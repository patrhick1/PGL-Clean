"""
Airtable Service Module
airtable_service.py
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
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from airtable_service import PodcastService # Avoid circular import for runtime

# Load .env variables to access your Airtable credentials
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
            logger.debug(
                f"Retrieved {len(records)} records using formula '{formula}'.")
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
                'Clients':
                self.api.table(self.base_id, 'Clients'),
                'Campaigns':
                self.api.table(self.base_id, 'Campaigns'),
                'Podcasts':
                self.api.table(self.base_id, 'Podcasts'),
                'Podcast_Episodes':
                self.api.table(self.base_id, 'Podcast_Episodes'),
                'Campaign Manager':
                self.api.table(self.base_id, 'Campaign Manager')
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
            raise ValueError(
                f"Table '{table_name}' does not exist in the base.")
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
            logger.error(
                f"Error retrieving record {record_id} from table '{table_name}': {e}"
            )
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
            logger.debug(
                f"Updated record {record_id} in table '{table_name}' with {fields}"
            )
            return updated_record
        except Exception as e:
            logger.error(
                f"Error updating record {record_id} in '{table_name}': {e}")
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
            logger.debug(
                f"Created new record in table '{table_name}' with {fields}")
            return new_record
        except Exception as e:
            logger.error(f"Error creating record in '{table_name}': {e}")
            return None

    def search_records(self, table_name, formula=None, view=None, max_records=None):
        """
        Search for records in a specified table using an Airtable filter formula.

        Args:
            table_name (str): The name of the table.
            formula (str): The Airtable filter formula (e.g., '{Name}="John"').
            view (str): (Optional) A specific view to search in.
            max_records (int): (Optional) The maximum number of records to return.

        Returns:
            list: A list of matching records.
        """
        try:
            table = self.get_table(table_name)
            params = {}
            if formula:
                params['formula'] = formula
            if view:
                params['view'] = view
            if max_records is not None:
                params['max_records'] = max_records
            records = table.all(**params)
            logger.debug(
                f"Found {len(records)} records in table '{table_name}' with formula '{formula}'"
            )
            return records
        except Exception as e:
            logger.error(
                f"Error searching records in '{table_name}' with formula '{formula}': {e}"
            )
            return []

    def create_records_batch(self, table_name, records_data):
        """
        Create multiple records in a table with a single API call.

        Args:
            table_name (str): The name of the table to add records to.
            records_data (list): A list of dictionaries, each containing fields for a new record.

        Returns:
            list: The created records or empty list on failure.
        """
        try:
            if not records_data:
                logger.warning(f"No records provided for batch creation in '{table_name}'")
                return []

            table = self.get_table(table_name)
            created_records = table.batch_create(records_data)
            logger.info(f"Successfully created {len(created_records)} records in '{table_name}' in batch")
            return created_records
        except Exception as e:
            logger.error(f"Error during batch creation in '{table_name}': {e}")
            return []

    def update_records_batch(self, table_name, records_data):
        """
        Update multiple records in a table with a single API call.

        Args:
            table_name (str): The name of the table to update records in.
            records_data (list): A list of dictionaries, each containing 'id' key with record ID 
                                and 'fields' key with a dictionary of fields to update.

        Returns:
            list: The updated records or empty list on failure.
        """
        try:
            if not records_data:
                logger.warning(f"No records provided for batch update in '{table_name}'")
                return []

            table = self.get_table(table_name)
            updated_records = table.batch_update(records_data)
            logger.info(f"Successfully updated {len(updated_records)} records in '{table_name}' in batch")
            return updated_records
        except Exception as e:
            logger.error(f"Error during batch update in '{table_name}': {e}")
            return []

    def delete_record(self, table_name, record_id):
        """
        Delete a single record from the specified table.

        Args:
            table_name (str): The name of the table.
            record_id (str): The ID of the record to delete.

        Returns:
            dict: A confirmation dictionary (e.g., {'deleted': True, 'id': record_id}) or None on failure.
        """
        try:
            table = self.get_table(table_name)
            result = table.delete(record_id)
            logger.info(f"Successfully deleted record {record_id} from table '{table_name}'.")
            return result
        except Exception as e:
            logger.error(f"Error deleting record {record_id} from '{table_name}': {e}")
            return None

    def delete_records_batch(self, table_name, record_ids):
        """
        Delete multiple records from a table with a single API call.

        Args:
            table_name (str): The name of the table to delete records from.
            record_ids (list): A list of record IDs (strings) to delete.

        Returns:
            list: A list of confirmation dictionaries for the deleted records or empty list on failure.
        """
        try:
            if not record_ids:
                logger.warning(f"No record IDs provided for batch deletion in '{table_name}'")
                return []

            table = self.get_table(table_name)
            deleted_records_info = table.batch_delete(record_ids)
            logger.info(f"Successfully processed batch delete for {len(deleted_records_info)} records in '{table_name}'.")
            # Note: Check the response content, as it might contain errors for specific IDs
            return deleted_records_info
        except Exception as e:
            logger.error(f"Error during batch deletion in '{table_name}': {e}")
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
            logger.debug(
                f"Retrieved {len(records)} records from view '{view}' in table '{table_name}'"
            )
            return records
        except Exception as e:
            logger.error(
                f"Error retrieving records from view '{view}' in '{table_name}': {e}"
            )
            return []


# --- Helper Functions --- 

def delete_old_podcast_episodes(podcast_service: 'PodcastService', months_ago=6, batch_size=10):
    """
    Finds and deletes records in the 'Podcast_Episodes' table published before
    a specified number of months ago.

    Args:
        podcast_service (PodcastService): An initialized PodcastService instance.
        months_ago (int): The age threshold in months. Defaults to 6.
        batch_size (int): The number of records to delete per API call. Defaults to 10.

    Returns:
        tuple: A tuple containing (number_of_records_deleted, number_of_errors).
    """
    table_name = 'Podcast_Episodes'
    date_field = 'Published' # Assuming this is the correct field name
    deleted_count = 0
    error_count = 0
    logger.info(f"--- Starting Deletion of Episodes Older Than {months_ago} Months ---")

    try:
        # 1. Search for old episodes
        search_formula = f"IS_BEFORE({{{date_field}}}, DATEADD(TODAY(), -{months_ago}, 'months'))"
        logger.info(f"Searching '{table_name}' with formula: {search_formula}")

        old_episodes = podcast_service.search_records(
            table_name=table_name,
            formula=search_formula
        )

        if old_episodes:
            logger.info(f"Found {len(old_episodes)} '{table_name}' record(s) published before {months_ago} months ago.")
            record_ids_to_delete = [record['id'] for record in old_episodes]
            logger.info(f"Attempting to delete {len(record_ids_to_delete)} records...")

            # 2. Delete records in batches
            for i in range(0, len(record_ids_to_delete), batch_size):
                batch_ids = record_ids_to_delete[i:i + batch_size]
                logger.info(f"Processing batch delete for {len(batch_ids)} IDs: {batch_ids}")
                try:
                    delete_results = podcast_service.delete_records_batch(table_name, batch_ids)
                    if delete_results:
                        successful_deletes = [res for res in delete_results if res.get('deleted')]
                        deleted_count += len(successful_deletes)
                        failed_deletes = len(batch_ids) - len(successful_deletes)
                        if failed_deletes > 0:
                            logger.error(f"  {failed_deletes} deletions failed within this batch.")
                            error_count += failed_deletes
                    else:
                         logger.error(f"  Batch delete call failed for IDs: {batch_ids}")
                         error_count += len(batch_ids)
                except Exception as batch_e:
                    logger.error(f"  Error during batch delete for IDs {batch_ids}: {batch_e}")
                    error_count += len(batch_ids)

            logger.info(f"Deletion process finished. Successfully deleted: {deleted_count}, Failed: {error_count}")

        else:
            logger.info(f"INFO: No '{table_name}' records found published before {months_ago} months ago. Nothing to delete.")

    except ValueError as ve:
         logger.error(f"Value Error during deletion process: {ve} - Check TABLE_NAME ('{table_name}')?")
         error_count = -1 # Indicate a search-level error
    except Exception as e:
        logger.error(f"General Error during deletion process: {e}")
        logger.error(f"Check if field '{date_field}' exists and is Date type in '{table_name}'.")
        error_count = -1 # Indicate a search-level error

    logger.info(f"--- Finished Deletion of Episodes Older Than {months_ago} Months ---")
    return deleted_count, error_count


def find_episodes_by_podcast_count(podcast_service: 'PodcastService', min_episode_count=10):
    """
    Finds records in 'Podcast_Episodes' linked to 'Podcasts' that have a total
    episode count greater than a specified minimum.

    Args:
        podcast_service (PodcastService): An initialized PodcastService instance.
        min_episode_count (int): The minimum episode count threshold. Defaults to 10.

    Returns:
        list: A list of found 'Podcast_Episodes' records, or an empty list if none found or error.
    """
    table_name = 'Podcast_Episodes'
    # Name of the Lookup field in Podcast_Episodes that shows the count from Podcasts
    count_lookup_field = 'EpCount'
    found_episodes = []
    logger.info(f"--- Starting Search for Episodes from Podcasts with > {min_episode_count} Episodes ---")

    try:
        # Construct the formula
        search_formula = f"{{{count_lookup_field}}} > {min_episode_count}"
        logger.info(f"Searching '{table_name}' with formula: {search_formula}")

        found_episodes = podcast_service.search_records(
            table_name=table_name,
            formula=search_formula
        )

        if found_episodes:
            logger.info(f"SUCCESS: Found {len(found_episodes)} '{table_name}' record(s) linked to podcasts with more than {min_episode_count} episodes.")
            # Optional: Log details of found records if needed
            # logger.info("Record IDs (showing first few):")
            # for i, record in enumerate(found_episodes[:20]): # Log first 20 IDs
            #     episode_count = record.get('fields', {}).get(count_lookup_field, 'N/A')
            #     podcast_link = record.get('fields', {}).get('Podcast', ['N/A'])[0] # Assuming 'Podcast' is the link field
            #     logger.info(f"  - ID: {record['id']}, Linked Podcast: {podcast_link}, EpCount Lookup: {episode_count}")
            # if len(found_episodes) > 20:
            #     logger.info(f"  ... (and {len(found_episodes) - 20} more)")
        else:
            logger.info(f"INFO: No '{table_name}' records found linked to podcasts with more than {min_episode_count} episodes.")

    except ValueError as ve:
         logger.error(f"Value Error during search: {ve} - Check TABLE_NAME ('{table_name}')?")
    except Exception as e:
        logger.error(f"General Error during search: {e}")
        logger.error(f"Check if field '{count_lookup_field}' exists and is Number/Lookup type in '{table_name}'.")

    logger.info(f"--- Finished Search for Episodes from Podcasts with > {min_episode_count} Episodes ---")
    return found_episodes


from collections import defaultdict
from datetime import datetime # For robust date handling if needed

def delete_excess_podcast_episodes(podcast_service: 'PodcastService', keep_count=10, batch_size=10):
    """
    Finds podcasts with more than 'keep_count' episodes and deletes the oldest
    episodes, keeping only the 'keep_count' most recent ones based on the
    'Published' date.

    Args:
        podcast_service (PodcastService): An initialized PodcastService instance.
        keep_count (int): The number of most recent episodes to keep per podcast. Defaults to 10.
        batch_size (int): The number of records to delete per API call. Defaults to 10.

    Returns:
        tuple: A tuple containing (number_of_records_deleted, number_of_errors).
    """
    table_name = 'Podcast_Episodes'
    count_lookup_field = 'EpCount' # Lookup field in Episodes showing total count for the Podcast
    podcast_link_field = 'Podcast' # Link field in Episodes linking to Podcasts table
    date_field = 'Published'       # Date field in Episodes used for sorting
    deleted_count = 0
    error_count = 0
    logger.info(f"--- Starting Deletion of Excess Episodes (Keep Top {keep_count}) ---")

    try:
        # 1. Fetch all episodes from podcasts potentially having excess episodes
        search_formula = f"{{{count_lookup_field}}} > {keep_count}"
        logger.info(f"Searching '{table_name}' for candidate episodes with formula: {search_formula}")
        candidate_episodes = podcast_service.search_records(
            table_name=table_name,
            formula=search_formula,
            # Ensure necessary fields are fetched if not default
            # fields=[podcast_link_field, date_field] # Add other fields if needed by logic
        )

        if not candidate_episodes:
            logger.info(f"No podcasts found with more than {{keep_count}} episodes based on lookup. Nothing to delete.")
            return 0, 0

        logger.info(f"Found {len(candidate_episodes)} candidate episodes. Grouping by Podcast...")

        # 2. Group episodes by Podcast ID
        episodes_by_podcast = defaultdict(list)
        for record in candidate_episodes:
            fields = record.get('fields', {})
            podcast_ids = fields.get(podcast_link_field)
            # Ensure it's a list and not empty
            if isinstance(podcast_ids, list) and podcast_ids:
                podcast_id = podcast_ids[0] # Assuming single link
                episodes_by_podcast[podcast_id].append(record)
            else:
                logger.warning(f"Skipping episode {record['id']} due to missing or invalid link in '{podcast_link_field}'.")

        # 3. Identify records to delete within each group
        record_ids_to_delete = []
        logger.info(f"Processing {len(episodes_by_podcast)} podcasts for potential excess episodes...")
        for podcast_id, episodes in episodes_by_podcast.items():
            if len(episodes) > keep_count:
                logger.debug(f"Podcast {podcast_id} has {len(episodes)} episodes (>{keep_count}). Sorting...")

                # Sort by 'Published' date (descending - newest first)
                # Handle missing dates - treat them as very old (put them at the end)
                def sort_key(record):
                    date_str = record.get('fields', {}).get(date_field)
                    if date_str:
                        try:
                            # Attempt to parse to ensure valid date comparison if needed,
                            # otherwise ISO strings usually sort correctly.
                            # Using epoch timestamp for robust comparison including None
                            return datetime.fromisoformat(date_str.replace('Z', '+00:00')).timestamp()
                        except (ValueError, TypeError):
                             return float('-inf') # Treat unparseable dates as oldest
                    return float('-inf') # Treat missing dates as oldest

                episodes.sort(key=sort_key, reverse=True) # Newest first

                # Identify episodes to delete (all beyond the keep_count)
                excess_episodes = episodes[keep_count:]
                excess_ids = [record['id'] for record in excess_episodes]
                record_ids_to_delete.extend(excess_ids)
                logger.debug(f"  Identified {len(excess_ids)} excess episodes for Podcast {podcast_id} to delete.")


        # 4. Batch delete
        if record_ids_to_delete:
            logger.info(f"Attempting to delete a total of {len(record_ids_to_delete)} excess records...")
            for i in range(0, len(record_ids_to_delete), batch_size):
                batch_ids = record_ids_to_delete[i:i + batch_size]
                logger.info(f"Processing batch delete for {len(batch_ids)} IDs: {batch_ids}")
                try:
                    delete_results = podcast_service.delete_records_batch(table_name, batch_ids)
                    if delete_results:
                        successful_deletes = [res for res in delete_results if res.get('deleted')]
                        deleted_count += len(successful_deletes)
                        failed_deletes = len(batch_ids) - len(successful_deletes)
                        if failed_deletes > 0:
                            logger.error(f"  {failed_deletes} deletions failed within this batch.")
                            error_count += failed_deletes
                    else:
                        logger.error(f"  Batch delete call failed for IDs: {batch_ids}")
                        error_count += len(batch_ids)
                except Exception as batch_e:
                    logger.error(f"  Error during batch delete for IDs {batch_ids}: {batch_e}")
                    error_count += len(batch_ids)
            logger.info(f"Deletion process finished. Successfully deleted: {deleted_count}, Failed: {error_count}")
        else:
            logger.info("No excess episodes found across all groups. Nothing to delete.")


    except ValueError as ve:
         logger.error(f"Value Error during deletion process: {ve} - Check TABLE_NAME ('{table_name}')?")
         error_count = -1 # Indicate a search-level error
    except Exception as e:
        logger.error(f"General Error during deletion process: {e}")
        logger.error(f"Check fields '{count_lookup_field}', '{podcast_link_field}', '{date_field}'.")
        error_count = -1 # Indicate a search-level error

    logger.info(f"--- Finished Deletion of Excess Episodes ---")
    return deleted_count, error_count


if __name__ == "__main__":
    logger.info("--- Running Airtable Service Main Block ---")
    load_dotenv()

    
    production_service = PodcastService()

    # --- Test the specific client name search formula ---
    logger.info("--- Testing Client Name Search Formula ---")
    
    # Define the client name to test with
    test_client_name = "Ashwin Ramesh"  # Replace with a known client name if different
    
    # Define the field name in Airtable that stores the client's name (as used in Campaign Manager)
    # This usually comes from a lookup or linked record.
    # As per campaign_status_tracker.py, it's "Client Name"
    airtable_cm_client_name_field = "Client Name" 
    
    # Construct the search formula
    # Escaping quotes within the client_name just in case, though usually not needed for typical names.
    escaped_client_name = test_client_name.replace('"', '\\"')
    search_formula = f"FIND(\"{escaped_client_name}\", ARRAYJOIN({{{airtable_cm_client_name_field}}})) > 0"
    
    logger.info(f"Searching 'Campaign Manager' for client '{test_client_name}' using formula: {search_formula}")
    
    try:
        client_specific_records = production_service.search_records(
            table_name='Campaign Manager',
            formula=search_formula
        )
        
        if client_specific_records:
            logger.info(f"Found {len(client_specific_records)} records for client '{test_client_name}'.")
            logger.info("First few records (or all if fewer than 5):")
            for i, record in enumerate(client_specific_records[:5]):
                # Try to display the 'Client Name' field from the record to verify
                client_name_from_record = record.get('fields', {}).get(airtable_cm_client_name_field, "Not Found")
                campaign_name_from_record = record.get('fields', {}).get("CampaignName", "Not Found")
                status_from_record = record.get('fields', {}).get("Status", "Not Found")
                logger.info(f"  Record ID: {record.get('id')}, Client Name Field: {client_name_from_record}, Campaign: {campaign_name_from_record}, Status: {status_from_record}")
        else:
            logger.info(f"No records found for client '{test_client_name}' using the formula.")
            
    except Exception as e:
        logger.error(f"Error during test search for client '{test_client_name}': {e}", exc_info=True)
        
    logger.info("--- Finished Testing Client Name Search Formula ---")

    # --- Test Fetching Episodes by Podcast ID --- # NEW TEST
    logger.info("--- Testing Fetching Episodes by Podcast ID ---")
    
    podcast_name = "Convergence"
    episodes_table_name = "Podcast_Episodes"
    podcast_link_field_name = "Podcast"
    
    # Construct the search formula using FIND and ARRAYJOIN for the linked record field
    episode_search_formula = f'{{{podcast_link_field_name}}} = "{podcast_name}"'
    
    logger.info(f"Searching '{episodes_table_name}' for podcast ID '{podcast_name}' using formula: {episode_search_formula}")
    
    try:
        linked_episodes = production_service.search_records(
            table_name=episodes_table_name,
            formula=episode_search_formula
        )
        
        if linked_episodes:
            logger.info(f"Found {len(linked_episodes)} episodes linked to podcast ID '{podcast_name}'.")
            logger.info("First few linked episodes (or all if fewer than 5):")
            for i, record in enumerate(linked_episodes[:5]):
                episode_title = record.get('fields', {}).get('Episode Title', 'No Title')
                episode_url = record.get('fields', {}).get('Episode URL', 'No URL')
                logger.info(f"  - Record ID: {record.get('id')}, Title: {episode_title}, URL: {episode_url}")
        else:
            logger.info(f"No episodes found linked to podcast ID '{podcast_name}' using the formula.")
            
    except Exception as e:
        logger.error(f"Error during test search for episodes linked to podcast ID '{podcast_name}': {e}", exc_info=True)
        
    logger.info("--- Finished Testing Fetching Episodes by Podcast ID ---")

    # You can keep or remove the older test below
    # logger.info("--- Original Test: Fetching first 2 CM records ---")
    # cm_records = production_service.search_records('Campaign Manager')
    # for record in cm_records[:2]:
    #     print(record)
    # logger.info("--- Finished Original Test ---")



