from flask_login import UserMixin
from . import login_manager, get_db_connection_from_app_config # Assuming get_db_connection is in __init__
from flask import current_app
from psycopg2.extras import DictCursor

class ClientUser(UserMixin):
    def __init__(self, campaign_id, username, client_name=None):
        self.id = campaign_id  # campaign_id will be used as the user ID for Flask-Login
        self.username = username # This is dashboard_username
        self.client_name = client_name if client_name else username

    # Flask-Login expects get_id to return a string
    def get_id(self):
        return str(self.id)

@login_manager.user_loader
def load_user(campaign_id):
    """Loads a user by their campaign_id (which Flask-Login will use as the user_id)."""
    conn = None
    try:
        conn = get_db_connection_from_app_config(current_app.config)
        if not conn:
            current_app.logger.error("load_user: Database connection failed.")
            return None
        
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("SELECT campaign_id, dashboard_username, client_name FROM Clients WHERE campaign_id = %s;", (str(campaign_id),))
            client_data = cur.fetchone()
        
        if client_data:
            return ClientUser(
                campaign_id=str(client_data['campaign_id']), 
                username=client_data['dashboard_username'],
                client_name=client_data['client_name']
            )
        return None
    except Exception as e:
        current_app.logger.error(f"Error in load_user for campaign_id {campaign_id}: {e}")
        return None
    finally:
        if conn:
            conn.close() 