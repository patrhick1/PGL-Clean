import psycopg2
from psycopg2 import sql
from psycopg2.extras import DictCursor
import os
from dotenv import load_dotenv

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

# --- Utility Functions ---
def execute_sql(conn, sql_statement, params=None):
    """Executes a given SQL statement."""
    try:
        with conn.cursor() as cur:
            cur.execute(sql_statement, params)
        conn.commit()
    except psycopg2.Error as e:
        print(f"Error executing SQL: {e}")
        if conn:
            conn.rollback()
        raise # Re-raise the exception to be handled by the caller

def create_timestamp_update_trigger_function(conn):
    """Creates or replaces a function to update a timestamp column to NOW()."""
    trigger_func_sql = """
    CREATE OR REPLACE FUNCTION update_modified_column()
    RETURNS TRIGGER AS $$
    BEGIN
       NEW.updated_at = NOW();
       RETURN NEW;
    END;
    $$ language 'plpgsql';
    """
    try:
        execute_sql(conn, trigger_func_sql)
        print("Timestamp update function 'update_modified_column' created/ensured.")
    except psycopg2.Error as e:
        print(f"Error creating timestamp update function: {e}")
        # Do not close connection here, let main handler do it


def apply_timestamp_update_trigger(conn, table_name):
    """Applies the timestamp update trigger to the specified table's updated_at column."""
    trigger_name = f"trigger_update_{table_name}_updated_at"
    apply_trigger_sql = sql.SQL("""
    DROP TRIGGER IF EXISTS {trigger_name} ON {table_name};
    CREATE TRIGGER {trigger_name}
    BEFORE UPDATE ON {table_name}
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();
    """).format(
        trigger_name=sql.Identifier(trigger_name),
        table_name=sql.Identifier(table_name)
    )
    try:
        execute_sql(conn, apply_trigger_sql)
        print(f"Timestamp update trigger applied to table '{table_name}'.")
    except psycopg2.Error as e:
        print(f"Error applying trigger to {table_name}: {e}")


# --- Table Creation Functions ---

def create_companies_table(conn):
    sql_statement = """
    CREATE TABLE IF NOT EXISTS COMPANIES (
        company_id UUID PRIMARY KEY,
        name TEXT,
        domain TEXT,
        description TEXT,
        category TEXT,
        primary_location TEXT,
        website_url TEXT,
        logo_url TEXT,
        employee_range INTEGER,
        est_arr NUMERIC,
        foundation_date DATE,
        twitter_handle TEXT,
        linkedin_url TEXT,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );
    """
    execute_sql(conn, sql_statement)
    print("Table COMPANIES created/ensured.")

def create_people_table(conn):
    sql_statement = """
    CREATE TABLE IF NOT EXISTS PEOPLE (
        person_id UUID PRIMARY KEY,
        company_id UUID REFERENCES COMPANIES(company_id) ON DELETE SET NULL,
        full_name TEXT,
        email TEXT UNIQUE,
        linkedin_profile_url TEXT,
        twitter_profile_url TEXT,
        instagram_profile_url TEXT,
        tiktok_profile_url TEXT,
        dashboard_username TEXT,
        dashboard_password_hash TEXT,
        attio_contact_id UUID,
        role VARCHAR(255),
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );
    """
    execute_sql(conn, sql_statement)
    print("Table PEOPLE created/ensured.")
    apply_timestamp_update_trigger(conn, "people")


def create_media_table(conn):
    sql_statement = """
    CREATE TABLE IF NOT EXISTS MEDIA (
        media_id SERIAL PRIMARY KEY,
        company_id UUID REFERENCES COMPANIES(company_id) ON DELETE SET NULL,
        name TEXT,
        rss_url TEXT,
        category TEXT,
        language VARCHAR(50),
        avg_downloads INTEGER,
        contact_email TEXT,
        fetched_episodes BOOLEAN,
        description TEXT,
        ai_description TEXT,
        embedding VECTOR(1536),
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_media_company_id ON MEDIA (company_id);
    CREATE INDEX IF NOT EXISTS idx_media_embedding_hnsw ON MEDIA USING hnsw (embedding vector_cosine_ops);
    """
    execute_sql(conn, sql_statement)
    print("Table MEDIA created/ensured.")

def create_media_people_table(conn):
    sql_statement = """
    CREATE TABLE IF NOT EXISTS MEDIA_PEOPLE (
        media_id INTEGER REFERENCES MEDIA(media_id) ON DELETE CASCADE,
        person_id UUID REFERENCES PEOPLE(person_id) ON DELETE CASCADE,
        show_role VARCHAR(255),
        host_confirmed BOOLEAN,
        linked_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (media_id, person_id)
    );
    """
    execute_sql(conn, sql_statement)
    print("Table MEDIA_PEOPLE created/ensured.")

def create_campaigns_table(conn):
    sql_statement = """
    CREATE TABLE IF NOT EXISTS CAMPAIGNS (
        campaign_id UUID PRIMARY KEY,
        person_id UUID REFERENCES PEOPLE(person_id) ON DELETE RESTRICT, -- Client/Owner
        attio_client_id UUID,
        campaign_name TEXT,
        campaign_type TEXT,
        campaign_bio TEXT,
        campaign_angles TEXT,
        embedding VECTOR(1536),
        start_date DATE,
        end_date DATE,
        goal_note TEXT,
        media_kit_url TEXT,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_campaigns_person_id ON CAMPAIGNS (person_id);
    CREATE INDEX IF NOT EXISTS idx_campaigns_embedding_hnsw ON CAMPAIGNS USING hnsw (embedding vector_cosine_ops);
    """
    execute_sql(conn, sql_statement)
    print("Table CAMPAIGNS created/ensured.")

def create_episodes_table(conn):
    sql_statement = """
    CREATE TABLE IF NOT EXISTS EPISODES (
        episode_id SERIAL PRIMARY KEY,
        media_id INTEGER REFERENCES MEDIA(media_id) ON DELETE CASCADE,
        title TEXT,
        publish_date DATE,
        duration_sec INTEGER,
        episode_summary TEXT,
        ai_episode_summary TEXT,
        episode_url TEXT,
        transcript TEXT,
        embedding VECTOR(1536),
        transcribe BOOLEAN,
        downloaded BOOLEAN,
        guest_names TEXT,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_episodes_media_id ON EPISODES (media_id);
    CREATE INDEX IF NOT EXISTS idx_episodes_embedding_hnsw ON EPISODES USING hnsw (embedding vector_cosine_ops);
    """
    execute_sql(conn, sql_statement)
    print("Table EPISODES created/ensured.")

def create_pitch_templates_table(conn):
    sql_statement = """
    CREATE TABLE IF NOT EXISTS PITCH_TEMPLATES (
        template_id TEXT PRIMARY KEY,
        media_type VARCHAR(100),
        target_media_type VARCHAR(100),
        language_code VARCHAR(10),
        tone VARCHAR(100),
        prompt_body TEXT,
        created_by TEXT,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );
    """
    execute_sql(conn, sql_statement)
    print("Table PITCH_TEMPLATES created/ensured.")

def create_pitch_generations_table(conn):
    sql_statement = """
    CREATE TABLE IF NOT EXISTS PITCH_GENERATIONS (
        pitch_gen_id SERIAL PRIMARY KEY,
        campaign_id UUID REFERENCES CAMPAIGNS(campaign_id) ON DELETE CASCADE,
        media_id INTEGER REFERENCES MEDIA(media_id) ON DELETE RESTRICT,
        template_id TEXT REFERENCES PITCH_TEMPLATES(template_id) ON DELETE RESTRICT,
        draft_text TEXT,
        ai_model_used TEXT,
        pitch_topic TEXT,
        temperature NUMERIC,
        generated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        reviewer_id TEXT,
        reviewed_at TIMESTAMPTZ,
        final_text TEXT,
        send_ready_bool BOOLEAN,
        generation_status VARCHAR(100)
    );
    CREATE INDEX IF NOT EXISTS idx_pitch_generations_campaign_id ON PITCH_GENERATIONS (campaign_id);
    CREATE INDEX IF NOT EXISTS idx_pitch_generations_media_id ON PITCH_GENERATIONS (media_id);
    CREATE INDEX IF NOT EXISTS idx_pitch_generations_template_id ON PITCH_GENERATIONS (template_id);
    """
    execute_sql(conn, sql_statement)
    print("Table PITCH_GENERATIONS created/ensured.")

def create_placements_table(conn): # Renamed from BOOKINGS
    sql_statement = """
    CREATE TABLE IF NOT EXISTS PLACEMENTS (
        placement_id SERIAL PRIMARY KEY,
        campaign_id UUID REFERENCES CAMPAIGNS(campaign_id) ON DELETE CASCADE,
        media_id INTEGER REFERENCES MEDIA(media_id) ON DELETE RESTRICT,
        current_status VARCHAR(100),
        status_ts TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        meeting_date DATE,
        call_date DATE,
        outreach_topic TEXT,
        recording_date DATE,
        go_live_date DATE,
        episode_link TEXT,
        notes TEXT,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_placements_campaign_id ON PLACEMENTS (campaign_id);
    CREATE INDEX IF NOT EXISTS idx_placements_media_id ON PLACEMENTS (media_id);
    """
    execute_sql(conn, sql_statement)
    print("Table PLACEMENTS created/ensured.")

def create_pitches_table(conn):
    sql_statement = """
    CREATE TABLE IF NOT EXISTS PITCHES (
        pitch_id SERIAL PRIMARY KEY,
        campaign_id UUID REFERENCES CAMPAIGNS(campaign_id) ON DELETE CASCADE,
        media_id INTEGER REFERENCES MEDIA(media_id) ON DELETE RESTRICT,
        attempt_no INTEGER,
        match_score NUMERIC,
        matched_keywords TEXT[],
        score_evaluated_at TIMESTAMPTZ,
        outreach_type VARCHAR(100),
        subject_line TEXT,
        body_snippet TEXT,
        send_ts TIMESTAMPTZ,
        reply_bool BOOLEAN,
        reply_ts TIMESTAMPTZ,
        pitch_gen_id INTEGER REFERENCES PITCH_GENERATIONS(pitch_gen_id) ON DELETE SET NULL,
        placement_id INTEGER REFERENCES PLACEMENTS(placement_id) ON DELETE SET NULL, -- Renamed from booking_id
        pitch_state VARCHAR(100),
        client_approval_status VARCHAR(100),
        created_by TEXT,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_pitches_campaign_id ON PITCHES (campaign_id);
    CREATE INDEX IF NOT EXISTS idx_pitches_media_id ON PITCHES (media_id);
    CREATE INDEX IF NOT EXISTS idx_pitches_pitch_gen_id ON PITCHES (pitch_gen_id);
    CREATE INDEX IF NOT EXISTS idx_pitches_placement_id ON PITCHES (placement_id);
    """
    execute_sql(conn, sql_statement)
    print("Table PITCHES created/ensured.")

def create_status_history_table(conn):
    sql_statement = """
    CREATE TABLE IF NOT EXISTS STATUS_HISTORY (
        status_history_id SERIAL PRIMARY KEY,
        placement_id INTEGER REFERENCES PLACEMENTS(placement_id) ON DELETE CASCADE, -- Renamed from booking_id
        old_status VARCHAR(100),
        new_status VARCHAR(100),
        changed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        changed_by TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_status_history_placement_id ON STATUS_HISTORY (placement_id);
    """
    execute_sql(conn, sql_statement)
    print("Table STATUS_HISTORY created/ensured.")

# --- Main Execution ---
def create_all_tables():
    """Creates all defined tables in the correct order."""
    conn = get_db_connection()
    if not conn:
        print("Database connection failed. Aborting table creation.")
        return

    try:
        # Create helper function for triggers first
        create_timestamp_update_trigger_function(conn)

        # Create tables in order of dependency
        create_companies_table(conn)
        create_people_table(conn) # Depends on COMPANIES (indirectly via trigger), applies trigger
        create_media_table(conn) # Depends on COMPANIES
        create_media_people_table(conn) # Depends on MEDIA, PEOPLE
        create_campaigns_table(conn) # Depends on PEOPLE
        create_episodes_table(conn) # Depends on MEDIA
        create_pitch_templates_table(conn)
        create_pitch_generations_table(conn) # Depends on CAMPAIGNS, MEDIA, PITCH_TEMPLATES
        create_placements_table(conn) # Depends on CAMPAIGNS, MEDIA
        create_pitches_table(conn) # Depends on CAMPAIGNS, MEDIA, PITCH_GENERATIONS, PLACEMENTS
        create_status_history_table(conn) # Depends on PLACEMENTS
        
        print("All tables checked/created successfully.")
    except psycopg2.Error as e:
        print(f"A database error occurred during table creation: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    print("Starting database schema creation process...")
    create_all_tables()
    print("Schema creation process finished.") 