import psycopg2
from psycopg2 import sql
from psycopg2.extras import DictCursor, Json
import os
from dotenv import load_dotenv
import json # For handling potential JSON string conversion if needed, though psycopg2 handles dicts for JSONB well.

# Load environment variables from .env file at the start
load_dotenv()

# --- Database Connection ---
def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname=os.environ.get("PGDATABASE"),
            user=os.environ.get("PGUSER"),
            password=os.environ.get("PGPASSWORD"),
            host=os.environ.get("PGHOST"),
            port=os.environ.get("PGPORT")
        )
        return conn
    except psycopg2.OperationalError as e:
        print(f"Error connecting to the database: {e}")
        print("Please ensure PostgreSQL is running and connection details are correct.")
        print("Ensure environment variables are set: PGDATABASE, PGUSER, PGPASSWORD, PGHOST, PGPORT")
        return None

# --- Table Creation ---
def create_clientsinstantlyleads_table():
    """Creates the clientsinstantlyleads table in the database if it doesn't exist."""
    conn = get_db_connection()
    if not conn:
        return

    create_table_sql = """
    CREATE TABLE IF NOT EXISTS clientsinstantlyleads (
        lead_id UUID PRIMARY KEY,
        timestamp_created TIMESTAMPTZ,
        timestamp_updated TIMESTAMPTZ,
        organization_id UUID,
        lead_status INTEGER,
        email_open_count INTEGER,
        email_reply_count INTEGER,
        email_click_count INTEGER,
        company_domain TEXT,
        status_summary JSONB,
        campaign_id UUID,
        email TEXT,
        personalization TEXT,
        website TEXT,
        last_name TEXT,
        first_name TEXT,
        company_name TEXT,
        phone TEXT,
        payload JSONB,
        status_summary_subseq JSONB,
        last_step_from TEXT,
        last_step_id UUID,
        last_step_timestamp_executed TIMESTAMPTZ,
        email_opened_step INTEGER,
        email_opened_variant INTEGER,
        email_replied_step INTEGER,
        email_replied_variant INTEGER,
        email_clicked_step INTEGER,
        email_clicked_variant INTEGER,
        lt_interest_status INTEGER,
        subsequence_id UUID,
        verification_status INTEGER,
        pl_value_lead TEXT,
        timestamp_added_subsequence TIMESTAMPTZ,
        timestamp_last_contact TIMESTAMPTZ,
        timestamp_last_open TIMESTAMPTZ,
        timestamp_last_reply TIMESTAMPTZ,
        timestamp_last_interest_change TIMESTAMPTZ,
        timestamp_last_click TIMESTAMPTZ,
        enrichment_status INTEGER,
        list_id UUID,
        last_contacted_from TEXT,
        uploaded_by_user UUID,
        upload_method TEXT,
        assigned_to UUID,
        is_website_visitor BOOLEAN,
        timestamp_last_touch TIMESTAMPTZ,
        esp_code INTEGER,
        backup_creation_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL
    );
    """
    # Comments for ENUM-like fields:
    # lead_status: 1:Active, 2:Paused, 3:Completed, -1:Bounced, -2:Unsubscribed, -3:Skipped
    # lt_interest_status: 0:OutOfOffice, 1:Interested, 2:MeetingBooked, 3:MeetingCompleted, 4:Closed, -1:NotInterested, -2:WrongPerson, -3:Lost
    # verification_status: 1:Verified, 11:Pending, 12:PendingVerificationJob, -1:Invalid, -2:Risky, -3:CatchAll, -4:JobChange
    # enrichment_status: 1:Enriched, 11:Pending, -1:NotAvailable, -2:Error
    # upload_method: 'manual', 'api', 'website-visitor'
    # esp_code: e.g., 0:InQueue, 1:Google, 2:Microsoft, 1000:NotFound

    create_email_index_sql = "CREATE INDEX IF NOT EXISTS idx_clientsinstantlyleads_email ON clientsinstantlyleads (email);"
    create_campaign_index_sql = "CREATE INDEX IF NOT EXISTS idx_clientsinstantlyleads_campaign_id ON clientsinstantlyleads (campaign_id);"
    create_ts_created_index_sql = "CREATE INDEX IF NOT EXISTS idx_clientsinstantlyleads_timestamp_created ON clientsinstantlyleads (timestamp_created);"
    create_payload_gin_index_sql = "CREATE INDEX IF NOT EXISTS idx_clientsinstantlyleads_payload_gin ON clientsinstantlyleads USING GIN (payload);"

    try:
        with conn.cursor() as cur:
            cur.execute(create_table_sql)
            cur.execute(create_email_index_sql)
            cur.execute(create_campaign_index_sql)
            cur.execute(create_ts_created_index_sql)
            cur.execute(create_payload_gin_index_sql)
            conn.commit()
            print("clientsinstantlyleads table checked/created successfully with indexes.")
    except psycopg2.Error as e:
        print(f"Error creating clientsinstantlyleads table or indexes: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

# --- CRUD Operations ---

def add_instantly_lead_record(lead_api_data):
    """Adds a new lead record from Instantly API to the clientsinstantlyleads table.
    lead_api_data should be a dictionary from the Instantly API response for a single lead.
    Returns the lead_id of the newly inserted record, or None on failure.
    """
    conn = get_db_connection()
    if not conn:
        return None

    # Map API data to table columns
    # Note: API keys are often camelCase or similar, table columns are snake_case
    # Ensure all required fields are present or handle missing data appropriately (e.g. default to None)
    
    # It's crucial that lead_api_data['id'] exists and is a valid UUID string for the PRIMARY KEY.
    if not lead_api_data.get('id'):
        print("Error: Lead data is missing 'id' field.")
        return None

    insert_data = {
        "lead_id": lead_api_data.get("id"),
        "timestamp_created": lead_api_data.get("timestamp_created"),
        "timestamp_updated": lead_api_data.get("timestamp_updated"),
        "organization_id": lead_api_data.get("organization"),
        "lead_status": lead_api_data.get("status"),
        "email_open_count": lead_api_data.get("email_open_count"),
        "email_reply_count": lead_api_data.get("email_reply_count"),
        "email_click_count": lead_api_data.get("email_click_count"),
        "company_domain": lead_api_data.get("company_domain"),
        "status_summary": lead_api_data.get("status_summary"), # psycopg2 handles dict -> JSONB
        "campaign_id": lead_api_data.get("campaign"),
        "email": lead_api_data.get("email"),
        "personalization": lead_api_data.get("personalization"),
        "website": lead_api_data.get("website"),
        "last_name": lead_api_data.get("last_name"),
        "first_name": lead_api_data.get("first_name"),
        "company_name": lead_api_data.get("company_name"),
        "phone": lead_api_data.get("phone"),
        "payload": lead_api_data.get("payload"), # psycopg2 handles dict -> JSONB
        "status_summary_subseq": lead_api_data.get("status_summary_subseq"), # psycopg2 handles dict -> JSONB
        "last_step_from": lead_api_data.get("last_step_from"),
        "last_step_id": lead_api_data.get("last_step_id"),
        "last_step_timestamp_executed": lead_api_data.get("last_step_timestamp_executed"),
        "email_opened_step": lead_api_data.get("email_opened_step"),
        "email_opened_variant": lead_api_data.get("email_opened_variant"),
        "email_replied_step": lead_api_data.get("email_replied_step"),
        "email_replied_variant": lead_api_data.get("email_replied_variant"),
        "email_clicked_step": lead_api_data.get("email_clicked_step"),
        "email_clicked_variant": lead_api_data.get("email_clicked_variant"),
        "lt_interest_status": lead_api_data.get("lt_interest_status"),
        "subsequence_id": lead_api_data.get("subsequence_id"),
        "verification_status": lead_api_data.get("verification_status"),
        "pl_value_lead": lead_api_data.get("pl_value_lead"),
        "timestamp_added_subsequence": lead_api_data.get("timestamp_added_subsequence"),
        "timestamp_last_contact": lead_api_data.get("timestamp_last_contact"),
        "timestamp_last_open": lead_api_data.get("timestamp_last_open"),
        "timestamp_last_reply": lead_api_data.get("timestamp_last_reply"),
        "timestamp_last_interest_change": lead_api_data.get("timestamp_last_interest_change"),
        "timestamp_last_click": lead_api_data.get("timestamp_last_click"),
        "enrichment_status": lead_api_data.get("enrichment_status"),
        "list_id": lead_api_data.get("list_id"),
        "last_contacted_from": lead_api_data.get("last_contacted_from"),
        "uploaded_by_user": lead_api_data.get("uploaded_by_user"),
        "upload_method": lead_api_data.get("upload_method"),
        "assigned_to": lead_api_data.get("assigned_to"),
        "is_website_visitor": lead_api_data.get("is_website_visitor"),
        "timestamp_last_touch": lead_api_data.get("timestamp_last_touch"),
        "esp_code": lead_api_data.get("esp_code")
        # backup_creation_timestamp has a DEFAULT
    }

    # Filter out None values to avoid inserting NULL for columns that might not exist in lead_api_data
    # or if you want to rely on table defaults for some fields (though most here don't have defaults other than backup_creation_timestamp)
    # However, for a backup, explicit NULLs for missing data from API might be desired.
    # The .get(key) method already returns None if key is not found, which psycopg2 handles as NULL.

    columns = list(insert_data.keys()) # Use list() to ensure it's a list for psycopg2
    # sql.Placeholder(col) creates named placeholders like %(col_name)s
    # So the second argument to execute should be a dictionary.
    values_placeholders = [sql.Placeholder(col) for col in columns]

    # --- UPSERT LOGIC --- 
    # Create the SET part for the ON CONFLICT DO UPDATE clause
    # Exclude lead_id from update, as it's the conflict target
    # Also, explicitly handle backup_creation_timestamp: don't update it on conflict, keep original backup time.
    update_columns = [col for col in columns if col not in ['lead_id', 'backup_creation_timestamp']]
    set_clause = sql.SQL(", ").join([
        sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(col), sql.Identifier(col))
        for col in update_columns
    ])
    # Always update a field like 'timestamp_updated' from the new data on conflict
    # and perhaps a new 'last_synced_timestamp' in your table to current time.
    # For now, we ensure all fields from EXCLUDED are updated except primary key and original backup time.

    insert_query = sql.SQL("""
        INSERT INTO clientsinstantlyleads ({columns})
        VALUES ({values})
        ON CONFLICT (lead_id)
        DO UPDATE SET {set_clause}, backup_creation_timestamp = clientsinstantlyleads.backup_creation_timestamp -- Keep original backup time
        RETURNING lead_id, (xmax = 0) AS inserted; -- xmax = 0 indicates an INSERT occurred
    """).format(
        columns=sql.SQL(', ').join(map(sql.Identifier, columns)),
        values=sql.SQL(', ').join(values_placeholders),
        set_clause=set_clause
    )
    
    # Prepare the dictionary for execute, wrapping dicts for JSONB columns with Json()
    execute_dict = {}
    for col, value in insert_data.items():
        if col in ["status_summary", "payload", "status_summary_subseq"] and isinstance(value, dict):
            execute_dict[col] = Json(value)
        else:
            execute_dict[col] = value

    try:
        with conn.cursor() as cur:
            cur.execute(insert_query, execute_dict) # Pass the dictionary directly
            result = cur.fetchone()
            inserted_lead_id = result[0]
            was_inserted = result[1] # True if INSERT, False if UPDATE
            conn.commit()
            if was_inserted:
                print(f"New lead record {inserted_lead_id} added to clientsinstantlyleads successfully.")
            else:
                print(f"Existing lead record {inserted_lead_id} updated in clientsinstantlyleads successfully.")
            return inserted_lead_id
    except psycopg2.IntegrityError as e: # Should be less common with ON CONFLICT
        print(f"Integrity error processing lead {lead_api_data.get('id')}: {e}")
        if conn:
            conn.rollback()
        return None
    except psycopg2.Error as e:
        print(f"Error adding lead {lead_api_data.get('id')} to backup: {e}")
        if conn:
            conn.rollback()
        return None
    finally:
        if conn:
            conn.close()

def update_instantly_lead_record(lead_id: str, update_data: dict):
    """Updates an existing lead record in the clientsinstantlyleads table.
    
    Args:
        lead_id (str): The UUID string of the lead to update.
        update_data (dict): A dictionary where keys are column names (snake_case)
                              and values are their new values.
                              
    Returns:
        bool: True if the update was successful and at least one row was affected, False otherwise.
    """
    conn = get_db_connection()
    if not conn:
        return False
    if not update_data:
        print("No update data provided.")
        return False

    # Prepare the SET clause dynamically
    set_clauses = []
    processed_update_data = {}

    for key, value in update_data.items():
        # Ensure the key is a valid column name to prevent SQL injection if keys come from untrusted source
        # For now, assuming keys are controlled and map to actual column names.
        set_clauses.append(sql.SQL("{} = {}").format(sql.Identifier(key), sql.Placeholder(key)))
        if key in ["status_summary", "payload", "status_summary_subseq"] and isinstance(value, dict):
            processed_update_data[key] = Json(value)
        else:
            processed_update_data[key] = value
    
    if not set_clauses:
        print("No valid fields to update.")
        return False

    # Add the lead_id to the dictionary for the WHERE clause placeholder
    processed_update_data['lead_id_where'] = lead_id

    update_query = sql.SQL("UPDATE clientsinstantlyleads SET {} WHERE lead_id = {} ").format(
        sql.SQL(', ').join(set_clauses),
        sql.Placeholder('lead_id_where')
    )

    try:
        with conn.cursor() as cur:
            cur.execute(update_query, processed_update_data)
            updated_rows = cur.rowcount # Number of rows affected
            conn.commit()
            if updated_rows > 0:
                print(f"Lead record {lead_id} updated successfully. {updated_rows} row(s) affected.")
                return True
            else:
                print(f"Lead record {lead_id} not found or no data changed. {updated_rows} row(s) affected.")
                return False # Could be True if no change is also success, but False if not found.
    except psycopg2.Error as e:
        print(f"Error updating lead {lead_id}: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def get_instantly_lead_by_id(lead_id: str):
    """Fetches a single lead record from clientsinstantlyleads by its lead_id."""
    conn = get_db_connection()
    if not conn:
        return None
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("SELECT * FROM clientsinstantlyleads WHERE lead_id = %s;", (lead_id,))
            record = cur.fetchone()
            return dict(record) if record else None
    except psycopg2.Error as e:
        print(f"Error fetching lead by ID {lead_id}: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_all_instantly_leads_for_campaign(campaign_id: str, limit: int = None, offset: int = 0):
    """Fetches all lead records for a specific campaign_id, with optional pagination."""
    conn = get_db_connection()
    if not conn:
        return []
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            query = "SELECT * FROM clientsinstantlyleads WHERE campaign_id = %s ORDER BY timestamp_created DESC"
            params = [campaign_id]
            if limit is not None:
                query += " LIMIT %s OFFSET %s"
                params.extend([limit, offset])
            query += ";"
            cur.execute(query, tuple(params))
            records = [dict(row) for row in cur.fetchall()]
            return records
    except psycopg2.Error as e:
        print(f"Error fetching leads for campaign {campaign_id}: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_all_instantly_leads(limit: int = None, offset: int = 0):
    """Fetches all lead records from clientsinstantlyleads, with optional pagination."""
    conn = get_db_connection()
    if not conn:
        return []
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            query = "SELECT * FROM clientsinstantlyleads ORDER BY timestamp_created DESC"
            params = []
            if limit is not None:
                query += " LIMIT %s OFFSET %s"
                params.extend([limit, offset])
            query += ";"
            cur.execute(query, tuple(params) if params else None)
            records = [dict(row) for row in cur.fetchall()]
            return records
    except psycopg2.Error as e:
        print(f"Error fetching all leads: {e}")
        return []
    finally:
        if conn:
            conn.close()

def delete_instantly_lead_record(lead_id: str):
    """Deletes a lead record from clientsinstantlyleads by its lead_id."""
    conn = get_db_connection()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM clientsinstantlyleads WHERE lead_id = %s;", (lead_id,))
            deleted_rows = cur.rowcount
            conn.commit()
            if deleted_rows > 0:
                print(f"Lead record {lead_id} deleted successfully.")
                return True
            else:
                print(f"Lead record {lead_id} not found for deletion.")
                return False
    except psycopg2.Error as e:
        print(f"Error deleting lead {lead_id}: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

# --- Example Usage (Placeholder - to be expanded) ---
if __name__ == "__main__":
    print("Instantly Leads DB Script (now clientsinstantlyleads)")
    print("-------------------------")

    create_clientsinstantlyleads_table()

    # --- Configuration for Backup ---
    all_campaign_ids = [
        "afe3a4d7-5ed7-4fd4-9f8f-cf4e2ddc843d",
        "d52f85c0-8341-42d8-9e07-99c6b758fa0b",
        "7b4a5386-8fa1-4059-8ded-398c0f48972b",
        "186fcab7-7c86-4086-9278-99238c453470",
        "ae1c1042-d10e-4cfc-ba4c-743a42550c85",
        "ccbd7662-bbed-46ee-bd8f-1bc374646472",
        "ad2c89bc-686d-401e-9f06-c6ff9d9b7430",
        "3816b624-2a1f-408e-91a9-b9f730d03e2b", # This one will be excluded
        "60346de6-915c-43fa-9dfa-b77983570359",
        "5b1053b5-8143-4814-a9dc-15408971eac8",
        "02b1d9ff-0afe-4b64-ac15-a886f43bdbce",
        "0725cdd8-b090-4da4-90af-6ca93ac3c267",
        "640a6822-c1a7-48c7-8385-63b0d4c283fc",
        "540b0539-f1c2-4612-94d8-df6fab42c2a7",
        "b55c61b6-262c-4390-b6e0-63dfca1620c2"
    ]

    excluded_campaign_id = "3816b624-2a1f-408e-91a9-b9f730d03e2b"
    
    campaign_ids_to_backup = [cid for cid in all_campaign_ids if cid != excluded_campaign_id]

    if not campaign_ids_to_backup:
        print("\nNo new campaign IDs specified for backup. Skipping Instantly API fetch and backup.")
    else:
        print(f"\n--- Starting Lead Backup for {len(campaign_ids_to_backup)} Campaign(s) ---")
        try:
            from src.external_api_service import InstantlyAPI 
        except ImportError:
            print("Error: Could not import InstantlyAPI from src.external_api_service.")
            print("Please ensure the file path is correct and the script is run from the PGL project root.")
            exit()

        instantly_service = InstantlyAPI()
        total_leads_fetched_all_campaigns = 0
        total_leads_added_all_campaigns = 0
        total_leads_failed_all_campaigns = 0

        for campaign_idx, current_campaign_id in enumerate(campaign_ids_to_backup):
            print(f"\nProcessing Campaign {campaign_idx + 1}/{len(campaign_ids_to_backup)}: ID {current_campaign_id}")
            
            print(f"Fetching all leads from Instantly campaign: {current_campaign_id}...")
            leads_from_api = instantly_service.list_leads_from_campaign(current_campaign_id)

            if leads_from_api:
                print(f"Fetched {len(leads_from_api)} leads from Instantly API for campaign {current_campaign_id}.")
                total_leads_fetched_all_campaigns += len(leads_from_api)
                current_campaign_added_count = 0
                current_campaign_failed_count = 0
                
                for i, lead_data in enumerate(leads_from_api):
                    # Minimal print to reduce console clutter for large numbers of leads
                    if (i + 1) % 100 == 0 or i == len(leads_from_api) - 1:
                         print(f"  Processing lead {i+1}/{len(leads_from_api)} for campaign {current_campaign_id}...")
                    
                    inserted_id = add_instantly_lead_record(lead_data)
                    if inserted_id:
                        current_campaign_added_count += 1
                    else:
                        current_campaign_failed_count += 1
                        # Detailed failure for a specific lead is printed within add_instantly_lead_record
                
                total_leads_added_all_campaigns += current_campaign_added_count
                total_leads_failed_all_campaigns += current_campaign_failed_count
                print(f"Backup for campaign {current_campaign_id} complete.")
                print(f"  Successfully added: {current_campaign_added_count} lead(s).")
                if current_campaign_failed_count > 0:
                    print(f"  Failed to add: {current_campaign_failed_count} lead(s).")

            elif isinstance(leads_from_api, list) and not leads_from_api:
                print(f"No leads found in campaign {current_campaign_id} or an API error occurred during fetching.")
            else:
                print(f"Could not fetch leads from API for campaign {current_campaign_id}. list_leads_from_campaign returned an unexpected type.")
        
        print("\n--- Overall Backup Summary ---")
        print(f"Total campaigns processed: {len(campaign_ids_to_backup)}")
        print(f"Total leads fetched from API across all campaigns: {total_leads_fetched_all_campaigns}")
        print(f"Total leads successfully added to backup: {total_leads_added_all_campaigns}")
        if total_leads_failed_all_campaigns > 0:
            print(f"Total leads failed to add to backup: {total_leads_failed_all_campaigns}")
    
    print("\nScript finished. Remember to configure your .env file for database and Instantly API access.") 