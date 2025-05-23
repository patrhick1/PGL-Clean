# src/attio_email_sent.py

from src.attio_service import AttioClient
from datetime import datetime
import os

# Initialize AttioClient. Assumes ATTIO_ACCESS_TOKEN is in environment variables.
# You might want to pass the API key directly if preferred: AttioClient(api_key="your_key")
attio_client = AttioClient()
PODCAST_OBJECT_SLUG = "podcast"

def update_attio_when_email_sent(data):
    """
    Updates Attio record when an email is sent via Instantly webhook.
    """
    airID = data.get('airID') # This is expected to be the Attio record_id for the podcast object

    if not airID:
        print("airID (Attio record_id) not provided in webhook data")
        return

    try:
        # Step 1: Get the podcast record from Attio using 'airID'
        podcast_record_response = attio_client.get_record(PODCAST_OBJECT_SLUG, airID)
        
        if not podcast_record_response or 'data' not in podcast_record_response:
            print(f"No Attio record found with id {airID} or error in response.")
            return

        podcast_data = podcast_record_response['data']
        current_attributes = podcast_data.get('values', {})
        
        current_description = current_attributes.get('description', '') # Assuming 'description' is the slug for Correspondence
        existing_outreach_date = current_attributes.get('outreach_date') # Assuming 'outreach_date' is the slug

        # Extract additional data from webhook payload
        timestamp = data.get('timestamp', datetime.now().isoformat())
        event_type = data.get('event_type', '') # e.g., "EMAIL_SENT"
        personalization = data.get('personalization', '') # Message content or summary

        # Append new correspondence entry
        # Ensure a clean separation for new entries
        new_entry_header = f"\n\n--- Entry: {timestamp} ---\nEvent Type: {event_type}\n"
        new_correspondence_detail = f"Message: {personalization}"
        
        # Check if current_description is a list (if it's a rich text field or similar) or string
        if isinstance(current_description, list):
            # Handle case where description might be structured (e.g., from rich text)
            # This example converts it to a string. Adjust if Attio returns structured text.
            current_description_text = "\n".join(current_description)
        else:
            current_description_text = current_description or ''

        updated_description = current_description_text + new_entry_header + new_correspondence_detail

        # Prepare fields to update
        today_date = datetime.now().strftime('%Y-%m-%d')
        attributes_to_update = {
            'description': updated_description,
            'outreach_date': today_date  # Set/update the outreach date
        }

        # If 'outreach_date' was not previously set, set 'relationship_stage' to 'Outreached'
        # Attio's 'get_record' returns actual values, so check if existing_outreach_date was None or empty
        if not existing_outreach_date:
            attributes_to_update['relationship_stage'] = 'Outreached' # Assuming 'relationship_stage' is the slug

        # Update the record in Attio
        updated_record = attio_client.update_record(
            object_type=PODCAST_OBJECT_SLUG,
            record_id=airID,
            attributes=attributes_to_update
        )

        if updated_record:
            print(f"Attio record {airID} updated successfully.")
        else:
            # The AttioClient._make_api_request method raises an exception on HTTP error,
            # so this 'else' block might not be reached if an API error occurs.
            # It's here for completeness if update_record could return None/False on logical failure
            # without raising an exception.
            print(f"Failed to update Attio record {airID}. Check AttioClient logs for details.")

    except Exception as e:
        print(f"An error occurred while updating Attio record {airID}: {str(e)}")
        # Potentially re-raise or handle more gracefully
        # raise

# Example Webhook Data Structure (for testing this function)
# data = {
#     'campaign_id': 'instantly_campaign_xyz', # Original script had this, might not be needed for Attio update if airID is direct
#     'airID': 'attio_record_id_123', # This should be the Attio record ID
#     'timestamp': '2023-10-26T10:00:00Z',
#     'event_type': 'EMAIL_SENT',
#     'personalization': 'Hi [Name], just checking in...'
# }
# update_attio_when_email_sent(data) 