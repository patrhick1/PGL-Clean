import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

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
        return None

if __name__ == "__main__":
    # This script is now for backing up the renamed table.
    # Original use for renaming is complete.
    print("Starting temporary backup script for 'clientsinstantlyleads'...")
    conn = get_db_connection()
    if not conn:
        print("Database connection failed. Backup aborted.")
        exit(1)

    # Updated table names
    source_table_name = "clientsinstantlyleads"
    backup_table_name = "clientsinstantlyleads_initialbackup" # Or a new timestamped name if preferred

    drop_sql = f"DROP TABLE IF EXISTS {backup_table_name};"
    backup_sql = f"CREATE TABLE {backup_table_name} AS TABLE {source_table_name} WITH DATA;"

    try:
        with conn.cursor() as cur:
            print(f"Executing: {drop_sql}")
            cur.execute(drop_sql)
            print(f"Dropped {backup_table_name} if it existed.")
            
            print(f"Executing: {backup_sql}")
            cur.execute(backup_sql)
            conn.commit()
            print(f"Successfully created {backup_table_name} as a copy of {source_table_name}.")
    except psycopg2.Error as e:
        print(f"Error during backup process: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
        print("Temporary backup script finished.") 