# client_dashboard_project/dashboard_flask_app/run.py
from app import create_app, Config

app = create_app(Config)

if __name__ == '__main__':
    # Note: FLASK_DEBUG from .env will control app.run(debug=...)
    # For production, use a proper WSGI server like Gunicorn or Waitress
    # and ensure FLASK_DEBUG is False.
    app.run(debug=app.config.get('DEBUG', False), host='0.0.0.0', port=int(app.config.get('PORT', 5001))) 