# client_dashboard_project/dashboard_flask_app/app/routes.py
from flask import Blueprint, render_template, redirect, url_for, flash, current_app
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import check_password_hash

from .forms import LoginForm
from .models import ClientUser # Assuming ClientUser is defined to load user data
from . import get_db_connection_from_app_config # From __init__.py
from psycopg2.extras import DictCursor

bp = Blueprint('main', __name__)

@bp.route('/')
@bp.route('/index')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    return redirect(url_for('main.login'))

@bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        username_to_check = form.username.data
        password_to_check = form.password.data
        conn = None
        try:
            conn = get_db_connection_from_app_config(current_app.config)
            if not conn:
                flash('Database connection error. Please try again later.', 'danger')
                return render_template('login.html', title='Login', form=form)

            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute("SELECT campaign_id, dashboard_username, dashboard_password_hash, client_name FROM Clients WHERE dashboard_username = %s;", (username_to_check,))
                client_record = cur.fetchone()
            
            if client_record and client_record['dashboard_password_hash'] and \
               check_password_hash(client_record['dashboard_password_hash'], password_to_check):
                
                user_obj = ClientUser(
                    campaign_id=str(client_record['campaign_id']), 
                    username=client_record['dashboard_username'],
                    client_name=client_record['client_name']
                )
                login_user(user_obj, remember=form.remember_me.data if 'remember_me' in form else False)
                flash(f'Welcome back, {user_obj.client_name}!', 'success')
                return redirect(url_for('main.dashboard'))
            else:
                flash('Invalid username or password. Please try again.', 'danger')
        except Exception as e:
            current_app.logger.error(f"Error during login for user {username_to_check}: {e}")
            flash('An unexpected error occurred. Please try again later.', 'danger')
        finally:
            if conn:
                conn.close()
                
    return render_template('login.html', title='Login', form=form)

@bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('main.login'))

@bp.route('/dashboard')
@login_required
def dashboard():
    # Later, this will fetch and pass analytics data to the template
    return render_template('dashboard.html', title='Dashboard') 