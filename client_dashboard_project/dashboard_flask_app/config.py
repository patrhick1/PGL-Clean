import os
from dotenv import load_dotenv

# Load environment variables from .env file in the dashboard_flask_app directory
# This assumes your .env file is located at client_dashboard_project/dashboard_flask_app/.env
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DOTENV_PATH = os.path.join(BASE_DIR, '.env')
if os.path.exists(DOTENV_PATH):
    load_dotenv(DOTENV_PATH)
else:
    print(f"Warning: .env file not found at {DOTENV_PATH}. Make sure it exists for database and secret key.")

class Config:
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY') or 'you-should-really-change-this-in-production'
    # Database connection details (ensure these are in your .env file for this app)
    PGDATABASE = os.environ.get('PGDATABASE')
    PGUSER = os.environ.get('PGUSER')
    PGPASSWORD = os.environ.get('PGPASSWORD')
    PGHOST = os.environ.get('PGHOST')
    PGPORT = os.environ.get('PGPORT')

    # Optional: SQLAlchemy configuration (if we choose to use it later)
    # SQLALCHEMY_DATABASE_URI = f"postgresql://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"
    # SQLALCHEMY_TRACK_MODIFICATIONS = False

    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true' 