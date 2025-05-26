# src/slack_service.py

import os
import requests
import json
import logging
from datetime import date
import html # Import the html module

logger = logging.getLogger(__name__)

# NEW: Import REPORT_ROW_ORDER from config.py
from .config import REPORT_ROW_ORDER

class SlackService:
    def __init__(self): # Corrected __init__ method name
        self.webhook_url = os.getenv('SLACK_WEBHOOK_URL')
        if not self.webhook_url:
            logger.error("SLACK_WEBHOOK_URL environment variable not set. Slack messages will not be sent.")
            # It's okay not to raise an error here, as the script might still function for sheets updates.

    def _send_slack_message(self, payload: dict):
        """Sends a message to Slack using the configured webhook URL."""
        if not self.webhook_url:
            logger.warning("Slack webhook URL is not configured. Skipping Slack message.")
            return False

        try:
            response = requests.post(
                self.webhook_url,
                data=json.dumps(payload),
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            logger.info("Slack message sent successfully.")
            return True
        except requests.exceptions.HTTPError as http_err: # More specific exception
            logger.error(f"Error sending Slack message: {http_err}")
            # Log the response content which often contains Slack's specific error
            if http_err.response is not None:
                logger.error(f"Slack API Response Status: {http_err.response.status_code}")
                logger.error(f"Slack API Response Body: {http_err.response.text}")
            return False
        except requests.exceptions.RequestException as e: # General request exception
            logger.error(f"Error sending Slack message (RequestException): {e}")
            return False

    def send_daily_report(self, client_reports: list[dict]):
        """
        Sends a consolidated daily report to Slack.

        :param client_reports: A list of dictionaries, where each dict contains:
                                - 'client_name': str
                                - 'weekly_metric_data': list (from _calculate_weekly_metrics)
                                - 'spreadsheet_url': str (optional, if you want to link)
        """
        if not client_reports:
            logger.info("No client reports to send to Slack.")
            return

        report_date = date.today().strftime("%Y-%m-%d")
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"📊 Daily Campaign Status Report - {report_date}"
                }
            },
            {
                "type": "divider"
            }
        ]

        for report in client_reports:
            client_name = html.escape(report['client_name']) # Escape client_name
            weekly_data = report['weekly_metric_data']
            raw_spreadsheet_url = report.get('spreadsheet_url')
            spreadsheet_url = html.escape(raw_spreadsheet_url) if raw_spreadsheet_url else None # Escape URL

            # Get data for the most recent week (last element in weekly_data)
            if not weekly_data:
                logger.warning(f"No weekly data for client {client_name}. Skipping Slack block for this client.")
                continue

            latest_week = weekly_data[-1] # Assuming weekly_data is chronological (oldest first)
            week_start = html.escape(latest_week['week_start'].strftime("%m/%d/%y")) # Escape dates
            week_end = html.escape(latest_week['week_end'].strftime("%m/%d/%y"))   # Escape dates
            status_counts = latest_week['status_counts']
            total_records = latest_week['total_records']

            # --- CONSOLIDATED BLOCK FOR EACH CLIENT ---
            # Build a single mrkdwn text string for the client's report
            client_report_text = f"*Client: {client_name}*\n_Week of {week_start} - {week_end}_\n"

            for row_info in REPORT_ROW_ORDER: # Use the imported REPORT_ROW_ORDER
                count = status_counts.get(row_info["source_status"], 0)
                client_report_text += f"• *{row_info['display_name']}:* {count}\n"

            client_report_text += f"• *Total Records (Changes) This Week:* {total_records}\n"

            if spreadsheet_url:
                client_report_text += f"🔗 <{spreadsheet_url}|View Full Report>\n"

            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": client_report_text
                }
            })
            # --- END CONSOLIDATED BLOCK ---

            blocks.append({"type": "divider"})

        # Remove the last divider if it's the last block
        if blocks and blocks[-1].get("type") == "divider":
            blocks.pop()

        payload = {
            "blocks": blocks
        }

        self._send_slack_message(payload)