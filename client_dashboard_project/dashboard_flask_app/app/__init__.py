# client_dashboard_project/dashboard_flask_app/app/__init__.py
from flask import Flask
from flask_login import LoginManager
from ..config import Config # Adjusted import path for sibling `config.py`
import psycopg2
from psycopg2.extras import DictCursor

# Initialize extensions
login_manager = LoginManager()
login_manager.login_view = 'main.login' # The route for the login page
login_manager.login_message_category = 'info' # Flash message category

def get_db_connection_from_app_config(app_config):
    """Establishes a database connection using app configuration."""
    try:
        conn = psycopg2.connect(
            dbname=app_config['PGDATABASE'],
            user=app_config['PGUSER'],
            password=app_config['PGPASSWORD'],
            host=app_config['PGHOST'],
            port=app_config['PGPORT']
        )
        return conn
    except psycopg2.OperationalError as e:
        app.logger.error(f"Error connecting to the database: {e}")
        return None
    except Exception as e:
        app.logger.error(f"A non-OperationalError occurred during DB connection: {e}")
        return None


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    login_manager.init_app(app)

    # Register blueprints here
    from .routes import bp as main_bp
    app.register_blueprint(main_bp)

    # Test DB connection on startup (optional, can be removed)
    # with app.app_context():
    #     conn = get_db_connection_from_app_config(app.config)
    #     if conn:
    #         app.logger.info("Database connection successful on app startup.")
    #         conn.close()
    #     else:
    #         app.logger.error("Database connection failed on app startup. Check config and .env file.")

    return app 