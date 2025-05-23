# src/attio_response.py

from src.attio_service import AttioClient
from datetime import datetime
import os

# Initialize AttioClient
attio_client = AttioClient()
PODCAST_OBJECT_SLUG = "podcast"

def update_correspondent_on_attio(data):
    """
    Updates the correspondence field and relationship stage in Attio when an email reply is received.
    """
    airID = data.get('airID') # Expected to be the Attio record_id
    timestamp = data.get('timestamp', datetime.now().isoformat())
    reply_text_snippet = data.get('reply_text_snippet')

    # Validate required fields
    if not airID or not reply_text_snippet:
        print("Webhook data missing required fields (airID or reply_text_snippet).")
        return

    try:
        # Fetch current record from Attio
        podcast_record_response = attio_client.get_record(PODCAST_OBJECT_SLUG, airID)
        if not podcast_record_response or 'data' not in podcast_record_response:
            print(f"Attio Record with ID {airID} not found or error in response.")
            return

        podcast_data = podcast_record_response['data']
        current_attributes = podcast_data.get('values', {})
        existing_description = current_attributes.get('description', '') # Map to 'description'

        # Append new correspondence
        new_entry_header = f"\n\n--- Reply Received: {timestamp} ---"
        new_reply_detail = f"Reply Snippet: {reply_text_snippet}"
        
        # Handle potential list format for description
        if isinstance(existing_description, list):
            existing_description_text = "\n".join(existing_description)
        else:
            existing_description_text = existing_description or ''
            
        new_description = existing_description_text + new_entry_header + new_reply_detail

        # Prepare fields to update
        attributes_to_update = {
            'description': new_description,
            'relationship_stage': 'Responded' # Update relationship stage
        }

        # Update the record in Attio
        updated_record = attio_client.update_record(
            object_type=PODCAST_OBJECT_SLUG,
            record_id=airID,
            attributes=attributes_to_update
        )

        if updated_record:
            print(f"Attio Record {airID} updated successfully with reply.")
        else:
            print(f"Failed to update Attio record {airID}. Check AttioClient logs.")

    except Exception as e:
        print(f"An error occurred while updating Attio record {airID} with reply: {str(e)}")
        # raise

# Example Webhook Data Structure (for testing)
# data = {
#     'airID': 'attio_record_id_123', # Attio record ID
#     'timestamp': '2023-10-26T11:00:00Z',
#     'reply_text_snippet': 'Thanks for reaching out! Let\'s chat.'
# }
# update_correspondent_on_attio(data) 