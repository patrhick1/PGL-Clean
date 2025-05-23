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
    print("Listing public tables in the database...")
    conn = get_db_connection()
    if not conn:
        print("Database connection failed. Cannot list tables.")
        exit(1)

    # SQL to list tables in the public schema (default schema)
    list_tables_sql = "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = 'public';"

    try:
        with conn.cursor() as cur:
            cur.execute(list_tables_sql)
            tables = cur.fetchall()
            if tables:
                print("Tables found:")
                for table in tables:
                    print(f"- {table[0]}")
            else:
                print("No tables found in the public schema.")
    except psycopg2.Error as e:
        print(f"Error listing tables: {e}")
    finally:
        if conn:
            conn.close()
        print("Finished listing tables.") 