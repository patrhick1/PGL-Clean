"""
Main Flask Application

This script runs two separate Flask apps: one for a simple dashboard and API 
(endpoints on port 5000), and another dedicated to receiving webhooks (endpoints 
on port 5001). The main Flask app provides endpoints to trigger various 
automations, while the webhook app receives data from Instantly (or any other 
email/webhook source) and updates Airtable accordingly.

Author: Paschal Okonkwor
Date: 2025-01-06
"""

import logging
import threading
from flask import Flask, render_template, request, jsonify

# Import the functions that handle specific tasks
from webhook_handler import poll_airtable_and_process, poll_podcast_search_database
from summary_guest_identification import process_summary_host_guest
from determine_fit import determine_fit
from pitch_episode_selection import pitch_episode_selection
from pitch_writer import pitch_writer
from send_pitch_to_instantly import send_pitch_to_instantly
from instantly_email_sent import update_airtable_when_email_sent
from instantly_response import update_correspondent_on_airtable

# -------------------------------------------------------------------
# Configure logging to show info in the console
# Adjust logging level as needed (DEBUG, INFO, WARNING, ERROR, CRITICAL)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# -------------------------------------------------------------------

# Create the Flask apps
main_app = Flask(__name__)
webhook_app = Flask(__name__)

def handle_automation(action):
    """
    Decide which automation to run based on the 'action' parameter. 
    Each action triggers a different background process to run.

    Args:
        action (str): The name of the automation to trigger (e.g., 'generate_bio_angles').

    Returns:
        str: A message describing which automation was triggered, or if the action is invalid.
    """
    try:
        if action == 'generate_bio_angles':
            poll_airtable_and_process()
            return "Bios and Angles generation automation triggered!"

        elif action == 'mipr_podcast_search':
            poll_podcast_search_database()
            return "MIPR Podcast Search automation triggered!"

        elif action == 'summary_host_guest':
            process_summary_host_guest()
            return "Summary, Host, and Guest Identification automation triggered!"

        elif action == 'determine_fit':
            determine_fit()
            return "Determine Fit automation triggered!"

        elif action == 'pitch_episode_angle':
            pitch_episode_selection()
            return "Pitch Episode and Angle Selection automation triggered!"

        elif action == 'pitch_writer':
            pitch_writer()
            return "Pitch Writer automation triggered!"

        elif action == 'send_pitch':
            send_pitch_to_instantly()
            return "Sending Pitch to Instantly automation triggered!"

        else:
            # If no matching action is found, we return a helpful message
            return "Invalid action!"
    except Exception as e:
        # Log and return error if something goes wrong
        logger.error(f"Error in handle_automation with action '{action}': {e}")
        return f"Error triggering automation for action '{action}'. Check logs."

@main_app.route('/')
def index():
    """
    Serves the HTML dashboard at the root ('/') route. 
    The template 'index.html' can display buttons or links to trigger each automation.
    """
    return render_template('index.html')

@main_app.route('/trigger/<action>', methods=['POST'])
def trigger_action(action):
    """
    This endpoint allows a user or service to POST to a URL like '/trigger/generate_bio_angles'
    and trigger the matching automation logic in handle_automation.

    Args:
        action (str): The portion of the URL specifying which automation to run.

    Returns:
        JSON response: A JSON-formatted message indicating success or error.
    """
    message = handle_automation(action)
    return jsonify({"message": message})

@webhook_app.route('/emailsent', methods=['POST'])
def webhook_emailsent():
    """
    Webhook endpoint that listens for a JSON payload indicating that an email was sent.
    This data is then used to update Airtable records via 'update_airtable_when_email_sent'.
    """
    data = request.json
    if not data:
        logger.warning("No JSON received at /emailsent webhook.")
        return jsonify({"error": "No JSON received"}), 400

    logger.info(f"/emailsent webhook data received: {data}")
    update_airtable_when_email_sent(data)
    return jsonify({"status": "success", "message": "Webhook processed!"}), 200

@webhook_app.route('/replyreceived', methods=['POST'])
def webhook_replyreceived():
    """
    Webhook endpoint that listens for a JSON payload indicating that a reply was received.
    This data is used to update Airtable records via 'update_correspondent_on_airtable'.
    """
    data = request.json
    if not data:
        logger.warning("No JSON received at /replyreceived webhook.")
        return jsonify({"error": "No JSON received"}), 400

    logger.info(f"/replyreceived webhook data received: {data}")
    update_correspondent_on_airtable(data)
    return jsonify({"status": "success", "message": "Webhook processed!"}), 200

def run_webhook():
    """
    Runs the webhook Flask app on port 5001 in a separate thread.
    This allows you to have one process for your main interface (port 5000)
    and another for receiving webhooks (port 5001).
    """
    logger.info("Starting webhook app on port 5001.")
    # debug=True is useful for development, but disable it in production
    # use_reloader=False prevents the Flask auto-restart from messing with the thread
    webhook_app.run(port=5001, debug=True, use_reloader=False)

def run_main_app():
    """
    Runs the main Flask app on port 5000. This is your main interface for triggering automations.
    """
    logger.info("Starting main app on port 5000.")
    main_app.run(port=5000, debug=True, use_reloader=False)

if __name__ == "__main__":
    # Start the webhook app in its own thread
    webhook_thread = threading.Thread(target=run_webhook)
    webhook_thread.start()

    # Then start the main Flask app
    run_main_app()
