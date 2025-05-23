# create_history_table.py

import os
import psycopg2
from psycopg2 import Error
from dotenv import load_dotenv
import logging
from datetime import datetime

# Import your Airtable service classes
from airtable_service import PodcastService, MIPRService

# Load environment variables
load_dotenv()

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- PostgreSQL Connection Details ---
DB_HOST = os.getenv('DB_HOST')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_PORT = os.getenv('DB_PORT')

# --- Database Table Name ---
HISTORY_TABLE_NAME = "airtable_status_history"

def get_db_connection():
    """Establishes and returns a connection to the PostgreSQL database."""
    conn = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        logger.debug("Successfully connected to PostgreSQL database.")
        return conn
    except Error as e:
        logger.error(f"Error connecting to PostgreSQL database: {e}")
        return None

def create_history_table():
    """
    Creates the 'airtable_status_history' table in the PostgreSQL database
    if it does not already exist.

    Table Fields:
    - id: SERIAL PRIMARY KEY (Auto-incrementing unique ID for each history record)
    - airtable_record_id: VARCHAR(18) NOT NULL (The unique ID of the Airtable record, e.g., 'recXYZ123')
    - airtable_table_name: VARCHAR(255) NOT NULL (The name of the Airtable table, e.g., 'Campaign Manager')
    - airtable_base_id: VARCHAR(17) NOT NULL (The ID of the Airtable base)
    - field_name: VARCHAR(255) NOT NULL (The name of the field that changed, e.g., 'Status')
    - old_value: TEXT (The previous value of the field)
    - new_value: TEXT NOT NULL (The new value of the field)
    - change_timestamp: TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP (When the change was recorded)
    - client_name: VARCHAR(255) (Denormalized client name for easier reporting)
    - campaign_name: VARCHAR(255) (Denormalized campaign name for easier reporting)
    - podcast_name: VARCHAR(255) (Denormalized podcast name for easier reporting, specific to 'Campaign Manager')
    - source_system: VARCHAR(100) DEFAULT 'Airtable' (Indicates where the change originated, e.g., 'Airtable', 'Manual')
    """
    conn = get_db_connection()
    if conn is None:
        return False

    try:
        cursor = conn.cursor()
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {HISTORY_TABLE_NAME} (
            id SERIAL PRIMARY KEY,
            airtable_record_id VARCHAR(18) NOT NULL,
            airtable_table_name VARCHAR(255) NOT NULL,
            airtable_base_id VARCHAR(17) NOT NULL,
            field_name VARCHAR(255) NOT NULL,
            old_value TEXT,
            new_value TEXT NOT NULL,
            change_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            client_name VARCHAR(255),
            campaign_name VARCHAR(255),
            podcast_name VARCHAR(255),
            source_system VARCHAR(100) DEFAULT 'Airtable'
        );
        """
        cursor.execute(create_table_sql)
        conn.commit()
        logger.info(f"Table '{HISTORY_TABLE_NAME}' created or already exists.")
        return True
    except Error as e:
        logger.error(f"Error creating table '{HISTORY_TABLE_NAME}': {e}")
        return False
    finally:
        if conn:
            conn.close()

def get_last_known_value(
    airtable_record_id: str,
    airtable_table_name: str,
    airtable_base_id: str,
    field_name: str
) -> str | None:
    """
    Retrieves the 'new_value' from the most recent history entry for a specific
    Airtable record and field. This will serve as the 'old_value' for the next change.

    Args:
        airtable_record_id (str): The Airtable record ID.
        airtable_table_name (str): The name of the Airtable table.
        airtable_base_id (str): The ID of the Airtable base.
        field_name (str): The name of the field (e.g., 'Status').

    Returns:
        str | None: The last known value of the field, or None if no history exists.
    """
    conn = get_db_connection()
    if conn is None:
        return None

    try:
        cursor = conn.cursor()
        select_sql = f"""
        SELECT new_value
        FROM {HISTORY_TABLE_NAME}
        WHERE airtable_record_id = %s
          AND airtable_table_name = %s
          AND airtable_base_id = %s
          AND field_name = %s
        ORDER BY change_timestamp DESC
        LIMIT 1;
        """
        cursor.execute(select_sql, (
            airtable_record_id, airtable_table_name, airtable_base_id, field_name
        ))
        result = cursor.fetchone()
        if result:
            logger.debug(f"Found last known value for {airtable_record_id}/{field_name}: '{result[0]}'")
            return result[0]
        else:
            logger.debug(f"No previous history found for {airtable_record_id}/{field_name}.")
            return None
    except Error as e:
        logger.error(f"Error retrieving last known value for {airtable_record_id}/{field_name}: {e}")
        return None
    finally:
        if conn:
            conn.close()

def insert_status_history(
    airtable_record_id: str,
    airtable_table_name: str,
    airtable_base_id: str,
    field_name: str,
    old_value: str,
    new_value: str,
    client_name: str = None,
    campaign_name: str = None,
    podcast_name: str = None,
    source_system: str = 'Airtable'
):
    """
    Inserts a new status change record into the 'airtable_status_history' table.

    Args:
        airtable_record_id (str): The Airtable record ID.
        airtable_table_name (str): The name of the Airtable table.
        airtable_base_id (str): The ID of the Airtable base.
        field_name (str): The name of the field that changed (e.g., 'Status').
        old_value (str): The previous value of the field.
        new_value (str): The new value of the field.
        client_name (str, optional): The client's name. Defaults to None.
        campaign_name (str, optional): The campaign's name. Defaults to None.
        podcast_name (str, optional): The podcast's name. Defaults to None.
        source_system (str, optional): The system that initiated the change. Defaults to 'Airtable'.
    """
    conn = get_db_connection()
    if conn is None:
        return

    try:
        cursor = conn.cursor()
        insert_sql = f"""
        INSERT INTO {HISTORY_TABLE_NAME} (
            airtable_record_id, airtable_table_name, airtable_base_id,
            field_name, old_value, new_value,
            client_name, campaign_name, podcast_name, source_system
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        cursor.execute(insert_sql, (
            airtable_record_id, airtable_table_name, airtable_base_id,
            field_value_to_string(field_name),
            field_value_to_string(old_value),
            field_value_to_string(new_value),
            field_value_to_string(client_name),
            field_value_to_string(campaign_name),
            field_value_to_string(podcast_name),
            source_system
        ))
        conn.commit()
        logger.info(f"Inserted status history for record {airtable_record_id}: '{old_value}' -> '{new_value}'")
    except Error as e:
        logger.error(f"Error inserting status history for record {airtable_record_id}: {e}")
    finally:
        if conn:
            conn.close()

def field_value_to_string(value):
    """
    Converts various Airtable field types (especially lookups which can be lists)
    into a single string for storage in TEXT fields.
    Handles None, lists, and other types.
    """
    if value is None:
        return None
    if isinstance(value, list):
        # Join list elements into a comma-separated string
        return ", ".join(str(item) for item in value)
    return str(value)


def main():
    """
    Main function to create the table and demonstrate logging a status change.
    """
    logger.info("Starting database table creation process...")
    if not create_history_table():
        logger.error("Failed to create history table. Exiting.")
        return

    logger.info("Demonstrating status change logging...")

    podcast_service = None
    try:
        podcast_service = PodcastService()
    except Exception as e:
        logger.error(f"Could not initialize PodcastService: {e}. Cannot demonstrate Airtable interaction.")
        return

    campaign_manager_table_name = 'Campaign Manager'
    # IMPORTANT: Replace this with a real record ID from your 'Campaign Manager' table
    # or use the search logic below to find one.
    example_record_id = "recBjyvznARcEL6Be" # Example from your provided data

    # --- Alternative: Find a record dynamically (uncomment to use) ---
    # records = podcast_service.search_records(campaign_manager_table_name, formula="{Status} = 'Prospect'", max_records=1)
    # if records:
    #     example_record_id = records[0]['id']
    #     logger.info(f"Found example record ID: {example_record_id}")
    # else:
    #     logger.warning(f"No 'Prospect' records found in '{campaign_manager_table_name}'. Please update example_record_id manually.")
    #     return
    # --- End Alternative ---

    if not example_record_id:
        logger.warning("No example record ID available for demonstration. Please set one manually or use the search logic.")
        return

    # 1. Fetch the current state of the record from Airtable
    airtable_record = podcast_service.get_record(campaign_manager_table_name, example_record_id)

    if airtable_record:
        # Extract current values from Airtable record
        current_status = airtable_record.get('fields', {}).get('Status', 'N/A')
        # Use field_value_to_string for denormalized fields to handle potential list types
        client_name = field_value_to_string(airtable_record.get('fields', {}).get('Client Name'))
        campaign_name = field_value_to_string(airtable_record.get('fields', {}).get('CampaignName'))
        podcast_name = field_value_to_string(airtable_record.get('fields', {}).get('Podcast')) # Assuming 'Podcast' is a single text field

        logger.info(f"Fetched Airtable record {example_record_id}. Current Status: '{current_status}'")

        # 2. Get the last known status from our PostgreSQL history table
        last_logged_status = get_last_known_value(
            airtable_record_id=example_record_id,
            airtable_table_name=campaign_manager_table_name,
            airtable_base_id=podcast_service.base_id,
            field_name='Status'
        )

        # Determine the old_value for the current log entry
        # If no history exists, consider the old_value as "Initial" or an empty string
        old_status_to_log = last_logged_status if last_logged_status is not None else "Initial"
        new_status_to_log = current_status

        # 3. Compare and log if there's a change
        if old_status_to_log != new_status_to_log:
            logger.info(f"Detected status change for {example_record_id}: '{old_status_to_log}' -> '{new_status_to_log}'. Logging...")
            insert_status_history(
                airtable_record_id=example_record_id,
                airtable_table_name=campaign_manager_table_name,
                airtable_base_id=podcast_service.base_id,
                field_name='Status',
                old_value=old_status_to_log,
                new_value=new_status_to_log,
                client_name=client_name,
                campaign_name=campaign_name,
                podcast_name=podcast_name,
                source_system='Automated Polling' # Or 'Airtable Webhook', 'Manual Update', etc.
            )
            # In a real scenario, if this script is *causing* the change, you'd update Airtable here.
            # If this script is *monitoring* changes (e.g., from webhooks or manual Airtable updates),
            # you would NOT update Airtable here.
            # Example if you were to update Airtable:
            # podcast_service.update_record(campaign_manager_table_name, example_record_id, {'Status': new_status_to_log})
            # logger.info(f"Airtable record {example_record_id} status updated to '{new_status_to_log}'.")
        else:
            logger.info(f"Status for {example_record_id} is still '{current_status}'. No change detected, no new history entry logged.")
            # Optionally, you could still log a "no change" entry if you want a heartbeat
            # insert_status_history(
            #     airtable_record_id=example_record_id,
            #     airtable_table_name=campaign_manager_table_name,
            #     airtable_base_id=podcast_service.base_id,
            #     field_name='Status',
            #     old_value=old_status_to_log,
            #     new_value=new_status_to_log,
            #     client_name=client_name,
            #     campaign_name=campaign_name,
            #     podcast_name=podcast_name,
            #     source_system='Status Check (No Change)'
            # )

    else:
        logger.warning(f"Airtable record {example_record_id} not found. Cannot demonstrate status change logging.")

    logger.info("Script finished.")

if __name__ == "__main__":
    main()