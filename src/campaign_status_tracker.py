# src/campaign_status_tracker.py

import os
from datetime import datetime, timedelta, date
from collections import Counter, defaultdict
import logging

from google.oauth2 import service_account
from googleapiclient.discovery import build
from dotenv import load_dotenv

from .google_sheets_service import GoogleSheetsService
from .db_utils import (
    create_history_table,
    get_client_status_history,
    get_all_client_names_from_history
)
from .slack_service import SlackService
# NEW: Import constants from config.py
from .config import (
    WEEKS_TO_REPORT,
    TRACKED_STATUSES,
    STATUSES_TO_IGNORE_IN_TOTAL,
    REPORT_ROW_ORDER,
    EXCLUDED_CLIENTS_FROM_REPORT
)

load_dotenv()

logger = logging.getLogger(__name__)

# --- Configuration ---
# Airtable Config (These are now conceptual references to the origin of data in the history table)
AIRTABLE_CAMPAIGN_MANAGER_TABLE = "Campaign Manager"
AIRTABLE_CM_CLIENT_NAME_FIELD = "Client Name"
AIRTABLE_CM_STATUS_FIELD = "Status"
AIRTABLE_CM_DATE_FIELD = "Last Modified" # This field is not directly used for fetching from DB, but conceptually represents the timestamp

# Google Drive/Sheets Config
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
CLIENT_SPREADSHEETS_TRACKING_FOLDER_ID = os.getenv('CLIENT_SPREADSHEETS_TRACKING_FOLDER_ID')
SHEETS_API_SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive.file'
]

# REMOVED: WEEKS_TO_REPORT, TRACKED_STATUSES, STATUSES_TO_IGNORE_IN_TOTAL, REPORT_ROW_ORDER
# They are now imported from src/config.py

class CampaignStatusTracker:
    def __init__(self):
        self.sheets_service = GoogleSheetsService()
        self.slack_service = SlackService()

        if not GOOGLE_APPLICATION_CREDENTIALS:
            logger.error("GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")
        if not CLIENT_SPREADSHEETS_TRACKING_FOLDER_ID:
            logger.error("CLIENT_SPREADSHEETS_TRACKING_FOLDER_ID environment variable not set.")
            raise ValueError("CLIENT_SPREADSHEETS_TRACKING_FOLDER_ID environment variable not set.")

        drive_credentials = service_account.Credentials.from_service_account_file(
            GOOGLE_APPLICATION_CREDENTIALS, scopes=SHEETS_API_SCOPES)
        self.drive_service = build('drive', 'v3', credentials=drive_credentials)
        logger.info("CampaignStatusTracker initialized with Sheets, Drive, and Slack services.")

    def _find_spreadsheet_in_folder(self, name, folder_id):
        """Finds a spreadsheet by name within a specific Google Drive folder."""
        safe_name = name.replace("'", "\\'")
        query = f"name = '{safe_name}' and mimeType = 'application/vnd.google-apps.spreadsheet' and '{folder_id}' in parents and trashed = false"
        try:
            response = self.drive_service.files().list(q=query, spaces='drive', fields='files(id, name, webViewLink)').execute()
            files = response.get('files', [])
            if files:
                logger.info(f"Found spreadsheet '{name}' with ID {files[0]['id']} in folder '{folder_id}'.")
                return files[0]['id'], files[0].get('webViewLink')
            return None, None
        except Exception as e:
            logger.error(f"Error finding spreadsheet '{name}' in folder '{folder_id}': {e}")
            return None, None

    def _get_or_create_spreadsheet_for_client(self, client_name):
        """Gets existing or creates new spreadsheet for the client."""
        spreadsheet_title = f"{client_name} - Campaign Status Tracker"
        spreadsheet_id, spreadsheet_url = self._find_spreadsheet_in_folder(spreadsheet_title, CLIENT_SPREADSHEETS_TRACKING_FOLDER_ID)

        if spreadsheet_id:
            return spreadsheet_id, spreadsheet_url
        else:
            logger.info(f"Creating new spreadsheet titled '{spreadsheet_title}'...")
            new_sheet_id = self.sheets_service.create_spreadsheet(title=spreadsheet_title)
            if new_sheet_id:
                logger.info(f"Spreadsheet created with ID: {new_sheet_id}. Moving to folder {CLIENT_SPREADSHEETS_TRACKING_FOLDER_ID}...")
                try:
                    file_metadata = self.drive_service.files().get(fileId=new_sheet_id, fields='parents, webViewLink').execute()
                    previous_parents = ",".join(file_metadata.get('parents', []))
                    self.drive_service.files().update(
                        fileId=new_sheet_id,
                        addParents=CLIENT_SPREADSHEETS_TRACKING_FOLDER_ID,
                        removeParents=previous_parents if previous_parents else None,
                        fields='id, parents, webViewLink'
                    ).execute()
                    # Re-fetch to ensure webViewLink is correct after move
                    updated_file_metadata = self.drive_service.files().get(fileId=new_sheet_id, fields='webViewLink').execute()
                    new_sheet_url = updated_file_metadata.get('webViewLink')

                    logger.info(f"Successfully moved spreadsheet {new_sheet_id} to folder {CLIENT_SPREADSHEETS_TRACKING_FOLDER_ID}.")
                    return new_sheet_id, new_sheet_url
                except Exception as e:
                    logger.error(f"Error moving spreadsheet {new_sheet_id} to folder {CLIENT_SPREADSHEETS_TRACKING_FOLDER_ID}: {e}")
                    return None, None
            else:
                logger.error(f"Failed to create spreadsheet for {client_name} via GoogleSheetsService.")
                return None, None

    def _parse_timestamp_to_date(self, timestamp_val):
        """Parses timestamp value (from DB, can be string or datetime object) to datetime.date object."""
        if not timestamp_val: return None
        try:
            if isinstance(timestamp_val, datetime):
                return timestamp_val.date()
            return datetime.fromisoformat(timestamp_val.replace('Z', '+00:00')).date()
        except ValueError:
            logger.warning(f"Warning: Could not parse timestamp value: {timestamp_val}")
            return None

    def _get_week_date_ranges(self, num_weeks):
        """Generates date ranges (Monday to Sunday) for the last num_weeks."""
        today = date.today()
        start_of_current_week = today - timedelta(days=today.weekday())

        date_ranges = []
        for i in range(num_weeks):
            week_start = start_of_current_week - timedelta(weeks=i)
            week_end = week_start + timedelta(days=6)
            date_ranges.append({"start": week_start, "end": week_end})

        return date_ranges[::-1] # Return in chronological order (oldest first)

    def _calculate_weekly_metrics(self, client_history_records, weekly_ranges):
        """Calculates status counts for each week for a specific client based on PostgreSQL history records."""
        weekly_data = []

        for week in weekly_ranges:
            week_start = week["start"]
            week_end = week["end"]

            status_counts = Counter()
            records_in_week = 0 # This will count status *changes* in the week

            for record in client_history_records:
                record_date_val = record.get('change_timestamp')
                status = record.get('new_value')

                if not record_date_val or not status: continue

                record_date = self._parse_timestamp_to_date(record_date_val)
                if not record_date: continue

                # Check if the record's change_timestamp falls within the current week range
                if week_start <= record_date <= week_end:
                    # Only count towards total if the NEW status is not in the ignore list
                    if status not in STATUSES_TO_IGNORE_IN_TOTAL:
                        records_in_week += 1

                    # Count for individual tracked statuses (based on the NEW status)
                    if status in TRACKED_STATUSES:
                        status_counts[status] += 1

            # Store counts for this week
            weekly_data.append({
                "week_start": week_start,
                "week_end": week_end,
                "status_counts": status_counts,
                "total_records": records_in_week
            })

        return weekly_data

    def _prepare_sheet_data(self, weekly_metric_data):
        """Formats the aggregated weekly data for writing to Google Sheets."""

        # Create headers
        headers = ["Metric / Status"] + [f"Week of {wd['week_start']:%m/%d/%y}" for wd in weekly_metric_data]
        data_rows = [headers]

        # Add rows for each metric/status defined in REPORT_ROW_ORDER
        for row_info in REPORT_ROW_ORDER:
            display_name = row_info["display_name"]
            source_status = row_info["source_status"]
            row_data = [display_name]
            for weekly_stats in weekly_metric_data:
                count = weekly_stats["status_counts"].get(source_status, 0)
                row_data.append(count)
            data_rows.append(row_data)

        # Add a gap for visual separation (optional)
        data_rows.append(["---"] * len(headers))

        # Add total row
        total_row = ["Total Records This Week"]
        for weekly_stats in weekly_metric_data:
            total_row.append(weekly_stats["total_records"])
        data_rows.append(total_row)

        return data_rows

    def update_single_client_spreadsheet(self, client_name: str) -> dict | None:
        """
        Updates the spreadsheet for a specific client by reading status changes from PostgreSQL.
        Returns a dictionary of report data for Slack, or None if failed.
        """
        logger.info(f"Starting spreadsheet update for client: {client_name}")

        client_history_records = get_client_status_history(client_name)

        if not client_history_records:
            logger.warning(f"No history records found for client '{client_name}' in PostgreSQL. Skipping spreadsheet update and Slack report for this client.")
            return None

        logger.info(f"Found {len(client_history_records)} history records for client '{client_name}'.")

        spreadsheet_id, spreadsheet_url = self._get_or_create_spreadsheet_for_client(client_name)
        if not spreadsheet_id:
            logger.error(f"Could not get or create spreadsheet for {client_name}. Aborting sheet update and Slack report for this client.")
            return None

        weekly_ranges = self._get_week_date_ranges(WEEKS_TO_REPORT)
        weekly_metric_data = self._calculate_weekly_metrics(client_history_records, weekly_ranges)

        if not weekly_metric_data:
            logger.warning(f"No weekly data calculated for {client_name}. Skipping sheet update and Slack report for this client.")
            return None

        sheet_data = self._prepare_sheet_data(weekly_metric_data)

        logger.info(f"Writing data to spreadsheet for {client_name} (Sheet ID: {spreadsheet_id})")
        try:
            self.sheets_service.write_sheet(spreadsheet_id, "Sheet1!A1", sheet_data)
            logger.info(f"Successfully updated spreadsheet for {client_name}.")
            
            # Prepare data for Slack report
            return {
                "client_name": client_name,
                "weekly_metric_data": weekly_metric_data,
                "spreadsheet_url": spreadsheet_url
            }
        except Exception as e:
            logger.error(f"Error writing to spreadsheet for {client_name} (ID: {spreadsheet_id}): {e}")
            return None

    def update_all_client_spreadsheets(self):
        """
        Main method to fetch client names from history, calculate weekly metrics,
        update sheets, and then send a consolidated Slack report.
        """
        logger.info("Starting full campaign status tracking update (Weekly Format) from PostgreSQL history...")

        client_names_from_history = get_all_client_names_from_history()

        if not client_names_from_history:
            logger.info("No client names found in history table. Exiting.")
            return

        # Filter out excluded clients
        included_clients = [name for name in client_names_from_history if name not in EXCLUDED_CLIENTS_FROM_REPORT]

        if not included_clients:
            logger.info("No clients to report on after exclusions. Exiting.")
            return

        logger.info(f"Processing {len(included_clients)} clients after excluding {len(client_names_from_history) - len(included_clients)} clients.")

        all_client_slack_reports = [] # List to collect reports for Slack

        # Iterate through each client and update/create spreadsheet
        for client_name in included_clients: # Iterate over the filtered list
            logger.info(f"\nProcessing client: {client_name}")
            client_report_data = self.update_single_client_spreadsheet(client_name)
            if client_report_data:
                all_client_slack_reports.append(client_report_data)

        logger.info("\nFull campaign status tracking update finished.")

        # Send consolidated Slack report
        if all_client_slack_reports:
            logger.info(f"Sending Slack report for {len(all_client_slack_reports)} clients.")
            self.slack_service.send_daily_report(all_client_slack_reports)
        else:
            logger.info("No successful client updates to report to Slack.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) # Ensure logging is configured for standalone run
    logger.info("Running Campaign Status Tracker script (Weekly Format) from PostgreSQL history...")
    
    logger.info("Attempting to create/verify database history table (airtable_status_history)...")
    if create_history_table():
        logger.info("Database history table check/creation successful.")
    else:
        logger.error("Failed to create/verify database history table. Report generation might fail if table doesn't exist or DB is inaccessible.")

    try:
        tracker = CampaignStatusTracker()
        tracker.update_all_client_spreadsheets() 
    except Exception as e:
        logger.error(f"An unhandled error occurred in CampaignStatusTracker: {e}", exc_info=True)