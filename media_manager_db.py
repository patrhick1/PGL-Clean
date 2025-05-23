import psycopg2
from psycopg2 import sql
from psycopg2.extras import DictCursor
import os # For environment variables
from dotenv import load_dotenv # Import load_dotenv

load_dotenv() # Load environment variables from .env file at the start

# --- Database Connection ---
def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        # Uses environment variables, which will be loaded from .env locally
        # or from Replit's secrets when deployed on Replit.
        conn = psycopg2.connect(
            dbname=os.environ.get("PGDATABASE"), # Changed from DB_NAME
            user=os.environ.get("PGUSER"),       # Changed from DB_USER
            password=os.environ.get("PGPASSWORD"), # Changed from DB_PASSWORD
            host=os.environ.get("PGHOST"),       # Changed from DB_HOST
            port=os.environ.get("PGPORT")        # Changed from DB_PORT
        )
        return conn
    except psycopg2.OperationalError as e:
        print(f"Error connecting to the database: {e}")
        print("Please ensure PostgreSQL is running and connection details are correct.")
        print("Ensure environment variables are set: PGDATABASE, PGUSER, PGPASSWORD, PGHOST, PGPORT")
        return None

# --- Table Creation ---
def create_media_manager_table():
    """Creates the MediaManager table in the database if it doesn't exist."""
    conn = get_db_connection()
    if not conn:
        return

    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS MediaManager (
                    id SERIAL PRIMARY KEY,
                    media_name TEXT NOT NULL,
                    host_name TEXT,
                    host_confirmed BOOLEAN,
                    email VARCHAR(255),
                    rss_feed TEXT,
                    status VARCHAR(100),
                    pitch_email TEXT,
                    interview_brief_initial TEXT,
                    call_date_initial TIMESTAMP,
                    placement_reach INTEGER,
                    download_proof_url TEXT,
                    live_link TEXT,
                    publish_date DATE,
                    client TEXT NOT NULL,
                    correspondence TEXT,
                    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    outreach_date DATE,
                    interview_brief_follow_up TEXT,
                    call_date_follow_up TIMESTAMP,
                    cost NUMERIC(12, 2),
                    cost_proof_url TEXT,
                    provided_package_media_kit_url TEXT,
                    reach_estimate INTEGER,
                    reach_estimate_proof_url TEXT,
                    responses TEXT,
                    last_contacted_date DATE,
                    description TEXT,
                    responded_date DATE
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_mediamanager_email ON MediaManager (email);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_mediamanager_media_name ON MediaManager (media_name);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_mediamanager_client ON MediaManager (client);
            """)
            conn.commit()
            print("MediaManager table checked/created successfully.")
    except psycopg2.Error as e:
        print(f"Error creating table: {e}")
        if conn:
            conn.rollback() # Rollback changes if error occurs
    finally:
        if conn:
            conn.close()

# --- CRUD Operations ---

def add_media_record(record_data):
    """Adds a new media record to the MediaManager table.
    record_data should be a dictionary with keys matching column names.
    Returns the ID of the newly inserted record, or None on failure.
    """
    conn = get_db_connection()
    if not conn:
        return None

    # Dynamically build the SQL query based on provided keys
    # Ensure all NOT NULL fields are present or have defaults
    required_fields = {"media_name", "client"}
    if not required_fields.issubset(record_data.keys()):
        print(f"Error: Missing required fields. Must include: {required_fields}")
        return None

    columns = record_data.keys()
    values = [record_data[column] for column in columns]

    insert_query = sql.SQL("INSERT INTO MediaManager ({}) VALUES ({}) RETURNING id").format(
        sql.SQL(', ').join(map(sql.Identifier, columns)),
        sql.SQL(', ').join(map(sql.Placeholder, columns))
    )

    try:
        with conn.cursor() as cur:
            cur.execute(insert_query, values)
            record_id = cur.fetchone()[0]
            conn.commit()
            print(f"Record added successfully with ID: {record_id}")
            return record_id
    except psycopg2.Error as e:
        print(f"Error adding record: {e}")
        if conn:
            conn.rollback()
        return None
    finally:
        if conn:
            conn.close()

def get_media_record_by_id(record_id):
    """Fetches a media record by its ID."""
    conn = get_db_connection()
    if not conn:
        return None
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur: # Use DictCursor for dictionary-like rows
            cur.execute("SELECT * FROM MediaManager WHERE id = %s;", (record_id,))
            record = cur.fetchone()
            return dict(record) if record else None
    except psycopg2.Error as e:
        print(f"Error fetching record by ID: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_all_media_records(limit=None, offset=0):
    """Fetches all media records, with optional pagination."""
    conn = get_db_connection()
    if not conn:
        return []
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            query = "SELECT * FROM MediaManager ORDER BY created_date DESC"
            if limit is not None:
                query += " LIMIT %s OFFSET %s;"
                cur.execute(query, (limit, offset))
            else:
                query += ";"
                cur.execute(query)
            records = [dict(row) for row in cur.fetchall()]
            return records
    except psycopg2.Error as e:
        print(f"Error fetching all records: {e}")
        return []
    finally:
        if conn:
            conn.close()

def update_media_record(record_id, update_data):
    """Updates an existing media record.
    update_data should be a dictionary of columns to update and their new values.
    Returns True on success, False on failure.
    """
    conn = get_db_connection()
    if not conn:
        return False
    if not update_data:
        print("No update data provided.")
        return False

    set_clause = sql.SQL(', ').join(
        sql.SQL("{} = {}").format(sql.Identifier(key), sql.Placeholder(key))
        for key in update_data.keys()
    )
    update_query = sql.SQL("UPDATE MediaManager SET {} WHERE id = {}").format(
        set_clause,
        sql.Placeholder("id")
    )
    
    values = update_data
    values['id'] = record_id

    try:
        with conn.cursor() as cur:
            cur.execute(update_query, values)
            updated_rows = cur.rowcount
            conn.commit()
            if updated_rows > 0:
                print(f"Record ID {record_id} updated successfully.")
                return True
            else:
                print(f"Record ID {record_id} not found or no changes made.")
                return False
    except psycopg2.Error as e:
        print(f"Error updating record: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def delete_media_record(record_id):
    """Deletes a media record by its ID."""
    conn = get_db_connection()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM MediaManager WHERE id = %s;", (record_id,))
            deleted_rows = cur.rowcount
            conn.commit()
            if deleted_rows > 0:
                print(f"Record ID {record_id} deleted successfully.")
                return True
            else:
                print(f"Record ID {record_id} not found.")
                return False
    except psycopg2.Error as e:
        print(f"Error deleting record: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

# --- Example Usage ---
if __name__ == "__main__":
    print("Media Manager Database Script")
    print("-----------------------------")

    # 1. Ensure the table exists
    create_media_manager_table()
    print("\n--- Running Examples ---")

    # 2. Add a new record
    print("\n1. Adding a new media record...")
    new_record_data_1 = {
        "media_name": "Tech Today Podcast",
        "host_name": "Alex Bell",
        "host_confirmed": True,
        "email": "alex.bell@techtoday.com",
        "rss_feed": "https://techtoday.com/rss",
        "status": "Pitched",
        "pitch_email": "Subject: Guest for Tech Today - AI Innovations...",
        "client": "FutureAI Corp",
        "outreach_date": "2024-05-01",
        "description": "Leading podcast on technology trends."
    }
    record_id_1 = add_media_record(new_record_data_1)

    new_record_data_2 = {
        "media_name": "Marketing Mavericks Show",
        "host_name": "Brenda Lee",
        "email": "brenda@marketingmavericks.biz",
        "status": "Initial Contact",
        "client": "GrowthHackers Inc.",
        "description": "Show about marketing strategies for startups."
    }
    record_id_2 = add_media_record(new_record_data_2)
    
    if record_id_1:
        print(f"Added record 1 with ID: {record_id_1}")
    if record_id_2:
        print(f"Added record 2 with ID: {record_id_2}")


    # 3. Get a record by ID
    if record_id_1:
        print(f"\n2. Fetching record with ID {record_id_1}...")
        record = get_media_record_by_id(record_id_1)
        if record:
            print(f"Found record: {record['media_name']} for client {record['client']}")
            # print(record) # Uncomment to see full record
        else:
            print(f"Record with ID {record_id_1} not found.")

    # 4. Get all records
    print("\n3. Fetching all media records...")
    all_records = get_all_media_records(limit=5) # Get latest 5
    if all_records:
        print(f"Found {len(all_records)} records:")
        for rec in all_records:
            print(f"  ID: {rec['id']}, Media: {rec['media_name']}, Client: {rec['client']}, Status: {rec['status']}")
    else:
        print("No records found.")

    # 5. Update a record
    if record_id_1:
        print(f"\n4. Updating status for record ID {record_id_1}...")
        update_success = update_media_record(record_id_1, {"status": "Interview Scheduled", "call_date_initial": "2024-06-15 10:00:00"})
        if update_success:
            updated_record = get_media_record_by_id(record_id_1)
            if updated_record:
                print(f"Updated record status: {updated_record['status']}, Call Date: {updated_record['call_date_initial']}")

    # 6. Delete a record
    if record_id_2: # Example: Delete the second record we created
        print(f"\n5. Deleting record ID {record_id_2}...")
        delete_success = delete_media_record(record_id_2)
        if delete_success:
            print(f"Record ID {record_id_2} was deleted.")
            # Verify deletion
            deleted_record_check = get_media_record_by_id(record_id_2)
            if not deleted_record_check:
                print(f"Verified: Record ID {record_id_2} no longer exists.")
            else:
                print(f"Verification Error: Record ID {record_id_2} still exists.")


    print("\n--- Examples Complete ---")
    print("Remember to replace placeholder connection details in get_db_connection().")
    print("And install psycopg2: pip install psycopg2-binary") 