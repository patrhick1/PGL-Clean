"""
Airtable Service Module

This module provides two classes (MIPRService and PodcastService) to simplify
interactions with Airtable. It can search, get, update, and create records
for given Airtable bases and tables.

Author: Paschal Okonkwor
Date: 2025-01-06
"""

import os
import logging
from pyairtable import Api
from dotenv import load_dotenv

# Load .env variables to access your Airtable credentials
load_dotenv()

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Retrieve your Airtable credentials from environment variables
AIRTABLE_API_KEY = os.getenv('AIRTABLE_PERSONAL_TOKEN')
MIPR_CRM_BASE_ID = os.getenv('MIPR_CRM_BASE_ID')
PODCAST_BASE_ID = os.getenv('PODCAST_BASE_ID')
CLIENT_TABLE_NAME = "Clients"

class MIPRService:
    """
    This class handles operations with a specific Airtable base (MIPR CRM).
    It can retrieve, filter, and update records in a specified 'Clients' table.
    """
    def __init__(self):
        """
        Initialize the MIPRService by connecting to Airtable using credentials 
        from the environment.
        """
        try:
            self.api_key = AIRTABLE_API_KEY
            self.base_id = MIPR_CRM_BASE_ID

            self.api = Api(self.api_key)
            self.client_table = self.api.table(self.base_id, CLIENT_TABLE_NAME)
            logger.info("MIPRService initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize MIPRService: {e}")
            raise

    def get_records_with_filter(self, formula):
        """
        Get records that match the given formula in the 'Clients' table.

        Args:
            formula (str): An Airtable filter formula.

        Returns:
            list: A list of matching records.
        """
        try:
            records = self.client_table.all(formula=formula)
            logger.debug(f"Retrieved {len(records)} records using formula '{formula}'.")
            return records
        except Exception as e:
            logger.error(f"Error getting records with filter '{formula}': {e}")
            return []

    def get_record_by_id(self, record_id):
        """
        Retrieve a record from the 'Clients' table by its record ID.

        Args:
            record_id (str): The Airtable record ID.

        Returns:
            dict: The record data or None on failure.
        """
        try:
            record = self.client_table.get(record_id)
            return record
        except Exception as e:
            logger.error(f"Error retrieving record {record_id}: {e}")
            return None

    def update_record(self, record_id, fields):
        """
        Update the record with the given ID in the 'Clients' table.

        Args:
            record_id (str): The ID of the record to update.
            fields (dict): A dictionary of field name-value pairs to update.

        Returns:
            dict: The updated record data or None on failure.
        """
        try:
            updated_record = self.client_table.update(record_id, fields)
            logger.debug(f"Record {record_id} updated with fields: {fields}")
            return updated_record
        except Exception as e:
            logger.error(f"Error updating record {record_id}: {e}")
            return None

class PodcastService:
    """
    This class handles operations for a separate Airtable base, which focuses on 
    podcast-related tables like 'Clients', 'Campaigns', and 'Podcasts'. 
    It provides utility methods to read, update, and create records in any table.
    """
    def __init__(self):
        """
        Initialize the PodcastService by connecting to Airtable using 
        the environment credentials for the podcast base.
        """
        try:
            self.api_key = AIRTABLE_API_KEY
            self.base_id = PODCAST_BASE_ID

            self.api = Api(self.api_key)

            # Store references to tables in a dictionary for easy access
            self.tables = {
                'Clients': self.api.table(self.base_id, 'Clients'),
                'Campaigns': self.api.table(self.base_id, 'Campaigns'),
                'Podcasts': self.api.table(self.base_id, 'Podcasts'),
                'Podcast_Episodes': self.api.table(self.base_id, 'Podcast_Episodes'),
                'Campaign Manager': self.api.table(self.base_id, 'Campaign Manager'),
                'Campaigns test': self.api.table(self.base_id, 'Campaigns test'),
            }
            logger.info("PodcastService initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize PodcastService: {e}")
            raise

    def get_table(self, table_name):
        """
        Retrieve a table object by name from the base.

        Args:
            table_name (str): Name of the table to fetch.

        Returns:
            pyairtable.Table: The table object.

        Raises:
            ValueError: If the table does not exist.
        """
        table = self.tables.get(table_name)
        if not table:
            logger.error(f"Table '{table_name}' does not exist.")
            raise ValueError(f"Table '{table_name}' does not exist in the base.")
        return table

    def get_record(self, table_name, record_id):
        """
        Retrieve a record by ID from a specified table.

        Args:
            table_name (str): The name of the table.
            record_id (str): The ID of the record to retrieve.

        Returns:
            dict: The record data.
        """
        try:
            table = self.get_table(table_name)
            record = table.get(record_id)
            return record
        except Exception as e:
            logger.error(f"Error retrieving record {record_id} from table '{table_name}': {e}")
            return None

    def update_record(self, table_name, record_id, fields):
        """
        Update a record in the specified table with new field values.

        Args:
            table_name (str): The name of the table.
            record_id (str): The ID of the record to update.
            fields (dict): A dictionary of fields to update.

        Returns:
            dict: The updated record data or None on failure.
        """
        try:
            table = self.get_table(table_name)
            updated_record = table.update(record_id, fields)
            logger.debug(f"Updated record {record_id} in table '{table_name}' with {fields}")
            return updated_record
        except Exception as e:
            logger.error(f"Error updating record {record_id} in '{table_name}': {e}")
            return None

    def create_record(self, table_name, fields):
        """
        Create a new record in a specified table.

        Args:
            table_name (str): The name of the table.
            fields (dict): A dictionary of fields to set in the new record.

        Returns:
            dict: The newly created record data or None on failure.
        """
        try:
            table = self.get_table(table_name)
            new_record = table.create(fields)
            logger.debug(f"Created new record in table '{table_name}' with {fields}")
            return new_record
        except Exception as e:
            logger.error(f"Error creating record in '{table_name}': {e}")
            return None

    def search_records(self, table_name, formula, view=None):
        """
        Search for records in a specified table using an Airtable filter formula.

        Args:
            table_name (str): The name of the table.
            formula (str): The Airtable filter formula (e.g., '{Name}="John"').
            view (str): (Optional) A specific view to search in.

        Returns:
            list: A list of matching records.
        """
        try:
            table = self.get_table(table_name)
            params = {'formula': formula}
            if view:
                params['view'] = view
            records = table.all(**params)
            logger.debug(f"Found {len(records)} records in table '{table_name}' with formula '{formula}'")
            return records
        except Exception as e:
            logger.error(f"Error searching records in '{table_name}' with formula '{formula}': {e}")
            return []

    def get_records_from_view(self, table_name, view):
        """
        Retrieve all records from a specific view in a table.

        Args:
            table_name (str): The name of the table.
            view (str): The name or ID of the view.

        Returns:
            list: A list of records from the specified view.
        """
        try:
            table = self.get_table(table_name)
            records = table.all(view=view)
            logger.debug(f"Retrieved {len(records)} records from view '{view}' in table '{table_name}'")
            return records
        except Exception as e:
            logger.error(f"Error retrieving records from view '{view}' in '{table_name}': {e}")
            return []

