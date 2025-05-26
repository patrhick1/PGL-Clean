# src/campaign_status_tracker.py

import os
from datetime import datetime, timedelta, date
from collections import Counter, defaultdict
import logging # Added for logging
import asyncio # Added for async operations if needed, though not directly used in this class's sync methods

from google.oauth2 import service_account
from googleapiclient.discovery import build
from dotenv import load_dotenv

from .airtable_service import PodcastService
from .google_sheets_service import GoogleSheetsService # Assuming this exists and works
from .db_utils import (
    create_history_table,
    get_last_known_value,
    insert_status_history,
    field_value_to_string # Ensure this is available if needed directly, or rely on insert_status_history
)

load_dotenv()

logger = logging.getLogger(__name__) # Initialize logger for this module

# --- Configuration ---
# Airtable Config
AIRTABLE_CAMPAIGN_MANAGER_TABLE = "Campaign Manager"
AIRTABLE_CM_CLIENT_NAME_FIELD = "Client Name" # Field containing the client's name
AIRTABLE_CM_STATUS_FIELD = "Status"         # Field for campaign status
AIRTABLE_CM_DATE_FIELD = "Last Modified"   # Date field to use for weekly grouping

# --- NEW: Airtable Base ID for history logging ---
# This should be the Base ID where the AIRTABLE_CAMPAIGN_MANAGER_TABLE resides.
# Assuming it's the same as PodcastService().base_id
AIRTABLE_BASE_ID = os.getenv('PODCAST_BASE_ID')
# --- END NEW ---

# Google Drive/Sheets Config
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
CLIENT_SPREADSHEETS_TRACKING_FOLDER_ID = os.getenv('CLIENT_SPREADSHEETS_TRACKING_FOLDER_ID')
SHEETS_API_SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive.file'
]

# --- Report Configuration ---
WEEKS_TO_REPORT = 5 # Current week + previous 4 weeks

# Statuses from Airtable needed for the report calculations
TRACKED_STATUSES = [
    "Outreached",
    "Responded",
    "Interested",
    "Pending Intro Call Booking",
    "Lost",
    "Form Submitted"
]

# Statuses to ignore when calculating "Total Records This Week"
STATUSES_TO_IGNORE_IN_TOTAL = [
    "Prospect",
    "OR Ready",
    "Fit",
    "Not a fit",
    "Episode and angles selected",
    "Pitch Done"
]

# Define the order and mapping for rows in the sheet
REPORT_ROW_ORDER = [
    {"display_name": "Messages sent", "source_status": "Outreached"},
    {"display_name": "Total replies", "source_status": "Responded"},
    {"display_name": "Positive replies", "source_status": "Interested"},
    {"display_name": "Form Submitted", "source_status": "Form Submitted"},
    {"display_name": "Meetings booked", "source_status": "Pending Intro Call Booking"},
    {"display_name": "Lost", "source_status": "Lost"},
]


class CampaignStatusTracker:
    def __init__(self):
        self.airtable_service = PodcastService()
        self.sheets_service = GoogleSheetsService()

        if not GOOGLE_APPLICATION_CREDENTIALS:
            logger.error("GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")
        if not CLIENT_SPREADSHEETS_TRACKING_FOLDER_ID:
            logger.error("CLIENT_SPREADSHEETS_TRACKING_FOLDER_ID environment variable not set.")
            raise ValueError("CLIENT_SPREADSHEETS_TRACKING_FOLDER_ID environment variable not set.")

        drive_credentials = service_account.Credentials.from_service_account_file(
            GOOGLE_APPLICATION_CREDENTIALS, scopes=SHEETS_API_SCOPES)
        self.drive_service = build('drive', 'v3', credentials=drive_credentials)
        logger.info("CampaignStatusTracker initialized with Airtable, Sheets, and Drive services.")

    def _find_spreadsheet_in_folder(self, name, folder_id):
        """Finds a spreadsheet by name within a specific Google Drive folder."""
        safe_name = name.replace("'", "\\'")
        query = f"name = '{safe_name}' and mimeType = 'application/vnd.google-apps.spreadsheet' and '{folder_id}' in parents and trashed = false"
        try:
            response = self.drive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
            files = response.get('files', [])
            if files:
                logger.info(f"Found spreadsheet '{name}' with ID {files[0]['id']} in folder '{folder_id}'.")
                return files[0]['id']
            return None
        except Exception as e:
            logger.error(f"Error finding spreadsheet '{name}' in folder '{folder_id}': {e}")
            return None

    def _get_or_create_spreadsheet_for_client(self, client_name):
        """Gets existing or creates new spreadsheet for the client."""
        spreadsheet_title = f"{client_name} - Campaign Status Tracker"
        spreadsheet_id = self._find_spreadsheet_in_folder(spreadsheet_title, CLIENT_SPREADSHEETS_TRACKING_FOLDER_ID)

        if spreadsheet_id:
            return spreadsheet_id
        else:
            logger.info(f"Creating new spreadsheet titled '{spreadsheet_title}'...")
            new_sheet_id = self.sheets_service.create_spreadsheet(title=spreadsheet_title)
            if new_sheet_id:
                logger.info(f"Spreadsheet created with ID: {new_sheet_id}. Moving to folder {CLIENT_SPREADSHEETS_TRACKING_FOLDER_ID}...")
                try:
                    file_metadata = self.drive_service.files().get(fileId=new_sheet_id, fields='parents').execute()
                    previous_parents = ",".join(file_metadata.get('parents', []))
                    self.drive_service.files().update(
                        fileId=new_sheet_id,
                        addParents=CLIENT_SPREADSHEETS_TRACKING_FOLDER_ID,
                        removeParents=previous_parents if previous_parents else None,
                        fields='id, parents'
                    ).execute()
                    logger.info(f"Successfully moved spreadsheet {new_sheet_id} to folder {CLIENT_SPREADSHEETS_TRACKING_FOLDER_ID}.")
                    return new_sheet_id
                except Exception as e:
                    logger.error(f"Error moving spreadsheet {new_sheet_id} to folder {CLIENT_SPREADSHEETS_TRACKING_FOLDER_ID}: {e}")
                    return None
            else:
                logger.error(f"Failed to create spreadsheet for {client_name} via GoogleSheetsService.")
                return None

    def _parse_airtable_datetime(self, datetime_str):
        """Parses Airtable datetime string to datetime.date object."""
        if not datetime_str: return None
        try:
            return datetime.fromisoformat(datetime_str.replace('Z', '+00:00')).date()
        except ValueError:
            logger.warning(f"Warning: Could not parse datetime string: {datetime_str}")
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

    def _calculate_weekly_metrics(self, client_records, weekly_ranges):
        """Calculates status counts for each week for a specific client."""
        weekly_data = []

        for week in weekly_ranges:
            week_start = week["start"]
            week_end = week["end"]

            status_counts = Counter()
            records_in_week = 0

            for record in client_records:
                fields = record.get('fields', {})
                record_date_str = fields.get(AIRTABLE_CM_DATE_FIELD)
                status = fields.get(AIRTABLE_CM_STATUS_FIELD)

                if not record_date_str or not status: continue

                record_date = self._parse_airtable_datetime(record_date_str)
                if not record_date: continue

                # Check if the record's date falls within the current week range
                if week_start <= record_date <= week_end:
                    # Only count towards total if status is not in the ignore list
                    if status not in STATUSES_TO_IGNORE_IN_TOTAL:
                        records_in_week += 1

                    # Count for individual tracked statuses
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

    def update_single_client_spreadsheet(self, client_name: str):
        """
        Updates the spreadsheet for a specific client and logs status changes to PostgreSQL.
        """
        logger.info(f"Starting spreadsheet update for client: {client_name}")

        # --- Retrieve client campaign records (existing logic) ---
        search_formula = f"FIND(\"{client_name.replace('"', '\\"')}\", ARRAYJOIN({{{AIRTABLE_CM_CLIENT_NAME_FIELD}}})) > 0"
        logger.info(f"Fetching records for client '{client_name}' using formula: {search_formula}")
        client_campaign_records = self.airtable_service.search_records(
            table_name=AIRTABLE_CAMPAIGN_MANAGER_TABLE,
            formula=search_formula
        )

        if not client_campaign_records:
            logger.warning(f"No records found for client '{client_name}' in Airtable. Skipping database history check and spreadsheet update.")
            return False # Return False as no operation was performed

        logger.info(f"Found {len(client_campaign_records)} records for client '{client_name}'.")

        # --- NEW: PostgreSQL History Logging ---
        if not AIRTABLE_BASE_ID:
            logger.error("AIRTABLE_BASE_ID is not configured. Cannot proceed with PostgreSQL history logging.")
        else:
            logger.info(f"Processing PostgreSQL history logs for {len(client_campaign_records)} records for client '{client_name}'...")
            for record in client_campaign_records:
                record_id = record.get('id')
                if not record_id:
                    logger.warning("Skipping record due to missing ID for PostgreSQL logging.")
                    continue

                current_fields = record.get('fields', {})
                current_status = field_value_to_string(current_fields.get(AIRTABLE_CM_STATUS_FIELD)) # Ensure it's a string

                if current_status is None: # Or handle empty string if that's possible
                    logger.debug(f"Record {record_id} has no current status. Skipping history log.")
                    continue
                
                # Get other denormalized fields (Client Name should be consistent here)
                # CampaignName and Podcast might vary per record, so fetch them from the record.
                campaign_name_val = field_value_to_string(current_fields.get('CampaignName'))
                podcast_name_val = field_value_to_string(current_fields.get('Podcast'))

                # Get last known status from PostgreSQL
                old_status = get_last_known_value(
                    airtable_record_id=record_id,
                    airtable_table_name=AIRTABLE_CAMPAIGN_MANAGER_TABLE,
                    airtable_base_id=AIRTABLE_BASE_ID, # Use the configured base ID
                    field_name=AIRTABLE_CM_STATUS_FIELD
                )

                if old_status is None:
                    # If no history, consider it "Initial" for the log's old_value
                    # or if the current status is the first one we're seeing.
                    old_status_for_log = "Initial"
                    # Log if current_status is not None and different from "Initial" (or always log if no history)
                    if current_status: # Only log if there's a status to log
                        logger.info(f"New record or first status for {record_id}: '{current_status}'. Logging to history.")
                        insert_status_history(
                            airtable_record_id=record_id,
                            airtable_table_name=AIRTABLE_CAMPAIGN_MANAGER_TABLE,
                            airtable_base_id=AIRTABLE_BASE_ID,
                            field_name=AIRTABLE_CM_STATUS_FIELD,
                            old_value=old_status_for_log,
                            new_value=current_status,
                            client_name=client_name, # Already known for this method call
                            campaign_name=campaign_name_val,
                            podcast_name=podcast_name_val,
                            source_system='Airtable Full Sync'
                        )
                elif old_status != current_status:
                    logger.info(f"Status change detected for record {record_id}: '{old_status}' -> '{current_status}'. Logging to history.")
                    insert_status_history(
                        airtable_record_id=record_id,
                        airtable_table_name=AIRTABLE_CAMPAIGN_MANAGER_TABLE,
                        airtable_base_id=AIRTABLE_BASE_ID,
                        field_name=AIRTABLE_CM_STATUS_FIELD,
                        old_value=old_status,
                        new_value=current_status,
                        client_name=client_name,
                        campaign_name=campaign_name_val,
                        podcast_name=podcast_name_val,
                        source_system='Airtable Full Sync'
                    )
                else:
                    logger.debug(f"No status change for record {record_id} ('{current_status}' == '{old_status}'). Not logging to history.")
        # --- END NEW PostgreSQL History Logging ---

        # --- Existing Google Sheet Update Logic ---
        spreadsheet_id = self._get_or_create_spreadsheet_for_client(client_name)
        if not spreadsheet_id:
            logger.error(f"Could not get or create spreadsheet for {client_name}. Aborting sheet update.")
            # Even if sheet update fails, history logging might have succeeded.
            # Consider if this should return False overall or if success of logging is also a factor.
            # For now, mirroring previous behavior: sheet failure means overall False for this method.
            return False

        weekly_ranges = self._get_week_date_ranges(WEEKS_TO_REPORT)
        weekly_metric_data = self._calculate_weekly_metrics(client_campaign_records, weekly_ranges)

        if not weekly_metric_data:
            logger.warning(f"No weekly data calculated for {client_name}. Skipping sheet update.")
            return False

        sheet_data = self._prepare_sheet_data(weekly_metric_data)

        logger.info(f"Writing data to spreadsheet for {client_name} (Sheet ID: {spreadsheet_id})")
        try:
            self.sheets_service.write_sheet(spreadsheet_id, "Sheet1!A1", sheet_data)
            logger.info(f"Successfully updated spreadsheet for {client_name}.")
            return True
        except Exception as e:
            logger.error(f"Error writing to spreadsheet for {client_name} (ID: {spreadsheet_id}): {e}")
            return False

    def update_all_client_spreadsheets(self):
        """
        Main method to fetch records, group by client, calculate weekly metrics, and update sheets.
        This method is for a full refresh, not for real-time webhook updates.
        """
        logger.info("Starting full campaign status tracking update (Weekly Format)...")

        all_campaign_records = self.airtable_service.search_records(
            table_name=AIRTABLE_CAMPAIGN_MANAGER_TABLE
        )

        if not all_campaign_records:
            logger.info(f"No records found in Airtable table '{AIRTABLE_CAMPAIGN_MANAGER_TABLE}'. Exiting.")
            return

        logger.info(f"Found {len(all_campaign_records)} total records. Grouping by client name...")

        records_by_client = defaultdict(list)
        for record in all_campaign_records:
            fields = record.get('fields', {})
            client_name_list = fields.get(AIRTABLE_CM_CLIENT_NAME_FIELD)
            client_name = None
            if isinstance(client_name_list, list) and client_name_list:
                potential_name = client_name_list[0]
                if isinstance(potential_name, str) and potential_name.strip():
                     client_name = potential_name.strip()
            if client_name:
                records_by_client[client_name].append(record)
            else:
                 logger.warning(f"Skipping record ID {record.get('id', 'N/A')} due to missing, invalid, or non-string client name in field '{AIRTABLE_CM_CLIENT_NAME_FIELD}'. Value: {client_name_list}")

        if not records_by_client:
            logger.info("No records with valid client names found after grouping. Exiting.")
            return

        logger.info(f"Processing {len(records_by_client)} clients found in campaign records.")

        # Get date ranges for the weeks to report
        weekly_ranges = self._get_week_date_ranges(WEEKS_TO_REPORT)
        logger.info(f"Reporting for weeks starting: {[r['start'] for r in weekly_ranges]}")

        # Iterate through each client group and update/create spreadsheet
        for client_name, client_records in records_by_client.items():
            logger.info(f"\nProcessing client: {client_name}")
            # Call the single client update method
            self.update_single_client_spreadsheet(client_name)

        logger.info("\nFull campaign status tracking update finished.")


if __name__ == "__main__":
    logger.info("Running Campaign Status Tracker script (Weekly Format)...")
    
    # Attempt to create the history table first
    logger.info("Attempting to create/verify database history table (airtable_status_history)...")
    if create_history_table(): # This function is from db_utils.py
        logger.info("Database history table check/creation successful.")
    else:
        logger.error("Failed to create/verify database history table. Webhook logging might fail if table doesn't exist or DB is inaccessible.")
        # Depending on how critical this is for a standalone run, you might decide
        # not to proceed if the table creation fails. For now, it logs an error and continues.

    try:
        tracker = CampaignStatusTracker()
        # This method will fetch all Campaign Manager records and update/create Google Sheets
        tracker.update_all_client_spreadsheets() 
    except Exception as e:
        logger.error(f"An unhandled error occurred in CampaignStatusTracker: {e}", exc_info=True)