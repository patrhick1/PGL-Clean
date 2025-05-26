import os
from datetime import datetime, timedelta, date
from collections import Counter, defaultdict
import logging
# Removed: import asyncio # Not used in this synchronous class

from google.oauth2 import service_account
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Removed: from airtable_service import PodcastService
from google_sheets_service import GoogleSheetsService
from db_utils import (
    create_history_table,
    get_client_status_history, # NEW: Function to get history for a specific client
    get_all_client_names_from_history # NEW: Function to get all unique client names from history
    # Removed: get_last_known_value, insert_status_history, field_value_to_string
)

load_dotenv()

logger = logging.getLogger(__name__)

# --- Configuration ---
# Airtable Config (These are now conceptual references to the origin of data in the history table)
AIRTABLE_CAMPAIGN_MANAGER_TABLE = "Campaign Manager"
AIRTABLE_CM_CLIENT_NAME_FIELD = "Client Name"
AIRTABLE_CM_STATUS_FIELD = "Status"
AIRTABLE_CM_DATE_FIELD = "Last Modified" # This field is not directly used for fetching from DB, but conceptually represents the timestamp

# Removed: AIRTABLE_BASE_ID as it was only used for the now-removed history logging in this script

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
        # Removed: self.airtable_service = PodcastService()
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
        logger.info("CampaignStatusTracker initialized with Sheets and Drive services.") # Updated log message

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

    def _parse_timestamp_to_date(self, timestamp_val): # Renamed function parameter
        """Parses timestamp value (from DB, can be string or datetime object) to datetime.date object."""
        if not timestamp_val: return None
        try:
            # If timestamp_val is already a datetime object (from psycopg2), convert it to date
            if isinstance(timestamp_val, datetime):
                return timestamp_val.date()
            # Otherwise, parse the string (e.g., if data was manually inserted or from a different source)
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

    def _calculate_weekly_metrics(self, client_history_records, weekly_ranges): # Changed parameter name
        """Calculates status counts for each week for a specific client based on PostgreSQL history records."""
        weekly_data = []

        for week in weekly_ranges:
            week_start = week["start"]
            week_end = week["end"]

            status_counts = Counter()
            records_in_week = 0 # This will count status *changes* in the week

            for record in client_history_records:
                # Data now comes directly from PostgreSQL history table
                record_date_val = record.get('change_timestamp')
                status = record.get('new_value')

                if not record_date_val or not status: continue

                record_date = self._parse_timestamp_to_date(record_date_val) # Use the renamed parser
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

    def update_single_client_spreadsheet(self, client_name: str):
        """
        Updates the spreadsheet for a specific client by reading status changes from PostgreSQL.
        """
        logger.info(f"Starting spreadsheet update for client: {client_name}")

        # --- Retrieve client campaign history records from PostgreSQL ---
        client_history_records = get_client_status_history(client_name)

        if not client_history_records:
            logger.warning(f"No history records found for client '{client_name}' in PostgreSQL. Skipping spreadsheet update.")
            return False

        logger.info(f"Found {len(client_history_records)} history records for client '{client_name}'.")

        # Removed: The entire PostgreSQL History Logging block, as this script no longer inserts data.

        # --- Existing Google Sheet Update Logic ---
        spreadsheet_id = self._get_or_create_spreadsheet_for_client(client_name)
        if not spreadsheet_id:
            logger.error(f"Could not get or create spreadsheet for {client_name}. Aborting sheet update.")
            return False

        weekly_ranges = self._get_week_date_ranges(WEEKS_TO_REPORT)
        # Pass the history records to the calculation method
        weekly_metric_data = self._calculate_weekly_metrics(client_history_records, weekly_ranges)

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
        Main method to fetch client names from history, calculate weekly metrics, and update sheets.
        """
        logger.info("Starting full campaign status tracking update (Weekly Format) from PostgreSQL history...") # Updated log message

        # Get all unique client names from the history table
        client_names_from_history = get_all_client_names_from_history()

        if not client_names_from_history:
            logger.info("No client names found in history table. Exiting.")
            return

        logger.info(f"Processing {len(client_names_from_history)} clients found in history records.")

        # Iterate through each client and update/create spreadsheet
        for client_name in client_names_from_history:
            logger.info(f"\nProcessing client: {client_name}")
            self.update_single_client_spreadsheet(client_name)

        logger.info("\nFull campaign status tracking update finished.")


if __name__ == "__main__":
    logger.info("Running Campaign Status Tracker script (Weekly Format) from PostgreSQL history...") # Updated log message
    
    # Attempt to create the history table first (still relevant if this script is run standalone
    # and the table might not exist, though its primary purpose is for the webhook to populate it)
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