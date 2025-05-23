# src/google_sheets_service.py

import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SERVICE_ACCOUNT_FILE = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

class GoogleSheetsService:
    def __init__(self):
        """Initializes the Google Sheets service client."""
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        self.sheets_service = build('sheets', 'v4', credentials=credentials)


    def read_sheet(self, spreadsheet_id, range_name):
        """Reads data from a specific range in a spreadsheet.

        Args:
            spreadsheet_id (str): The ID of the spreadsheet.
            range_name (str): The A1 notation of the range to read (e.g., 'Sheet1!A1:B2').

        Returns:
            list: A list of lists containing the data read from the sheet.
        """
        result = self.sheets_service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id, range=range_name).execute()
        values = result.get('values', [])
        return values

    def create_spreadsheet(self, title):
        spreadsheet = {
            'properties': {'title': title}
        }
        try:
            spreadsheet = self.sheets_service.spreadsheets().create(body=spreadsheet, fields='spreadsheetId').execute()
            logger.info(f"Spreadsheet created: {spreadsheet.get('spreadsheetId')}")
            return spreadsheet.get('spreadsheetId')
        except Exception as e:
            logger.error(f"Error creating spreadsheet '{title}': {e}")
            return None

    def write_sheet(self, spreadsheet_id, range_name, values):
        body = {
            'values': values
        }
        try:
            result = self.sheets_service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id, range=range_name,
                valueInputOption='RAW', body=body).execute()
            logger.info(f"{result.get('updatedCells')} cells updated in {spreadsheet_id} at {range_name}.")
            return result
        except Exception as e:
            logger.error(f"Error writing to sheet {spreadsheet_id} at {range_name}: {e}")
            raise # Re-raise to be caught by the caller

    def append_sheet(self, spreadsheet_id, range_name, values):
        """Appends data to a table within a sheet. Finds the first empty row.

        Args:
            spreadsheet_id (str): The ID of the spreadsheet.
            range_name (str): The A1 notation of the table range (e.g., 'Sheet1!A1').
                             The method appends after the last row of this table.
            values (list): A list of lists containing the data to append.
        """
        body = {
            'values': values
        }
        result = self.sheets_service.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id, range=range_name,
            valueInputOption='USER_ENTERED',
            insertDataOption='INSERT_ROWS',
            body=body).execute()
        print(f"Appended data to range: {result.get('updates').get('updatedRange')}")
        return result 