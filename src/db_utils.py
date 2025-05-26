# db_utils.py (additions only)

import os
import psycopg2
from psycopg2 import Error
import logging
from dotenv import load_dotenv
from datetime import datetime # Added for type hinting and potential datetime object handling

# Load environment variables (ensure this is called where needed, e.g., in main_fastapi.py)
load_dotenv()

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


# --- NEW FUNCTIONS FOR REPORTING ---

def get_client_status_history(client_name: str) -> list[dict]:
    """
    Retrieves all status history entries for a specific client from the PostgreSQL table.
    """
    conn = get_db_connection()
    if conn is None:
        return []

    records = []
    try:
        cursor = conn.cursor()
        select_sql = f"""
        SELECT
            id, airtable_record_id, airtable_table_name, airtable_base_id,
            field_name, old_value, new_value, change_timestamp,
            client_name, campaign_name, podcast_name, source_system
        FROM {HISTORY_TABLE_NAME}
        WHERE client_name = %s
        ORDER BY change_timestamp ASC;
        """
        cursor.execute(select_sql, (client_name,))
        columns = [desc[0] for desc in cursor.description]
        for row in cursor.fetchall():
            records.append(dict(zip(columns, row)))
        logger.info(f"Retrieved {len(records)} history records for client '{client_name}'.")
        return records
    except Error as e:
        logger.error(f"Error retrieving history for client '{client_name}': {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_all_client_names_from_history() -> list[str]:
    """
    Retrieves all unique client names from the airtable_status_history table.
    """
    conn = get_db_connection()
    if conn is None:
        return []

    client_names = []
    try:
        cursor = conn.cursor()
        select_sql = f"""
        SELECT DISTINCT client_name
        FROM {HISTORY_TABLE_NAME}
        WHERE client_name IS NOT NULL AND client_name != '';
        """
        cursor.execute(select_sql)
        for row in cursor.fetchall():
            client_names.append(row[0])
        logger.info(f"Retrieved {len(client_names)} unique client names from history.")
        return client_names
    except Error as e:
        logger.error(f"Error retrieving unique client names from history: {e}")
        return []
    finally:
        if conn:
            conn.close()