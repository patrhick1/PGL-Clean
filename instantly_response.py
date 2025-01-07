# app/instantly_response.py

from airtable_service import PodcastService

def update_correspondent_on_airtable(data):
    """
    Updates the Correspondence field in Airtable with the new reply.
    """
    # Extract data from webhook payload
    airID = data.get('airID')
    timestamp = data.get('timestamp')
    reply_text_snippet = data.get('reply_text_snippet')

    # Validate required fields
    if not airID or not timestamp or not reply_text_snippet:
        print("Webhook data missing required fields.")
        return

    # Initialize Airtable service
    airtable_service = PodcastService()

    CAMPAIGN_MANAGER_TABLE_NAME = 'Campaign Manager'  


    # Fetch current record from Airtable
    campaign_manager_record = airtable_service.get_record(CAMPAIGN_MANAGER_TABLE_NAME, airID)
    if not campaign_manager_record:
        print(f"Record with ID {airID} not found in Airtable.")
        return

    # Get existing Correspondence
    campaign_manager_fields = campaign_manager_record.get('fields', {})
    existing_correspondence = campaign_manager_fields.get('Correspondence', '')

    # Append new correspondence
    new_correspondence = f"{existing_correspondence}\n\n{timestamp}\n{reply_text_snippet}"

    # Prepare fields to update
    fields_to_update = {
        'Correspondence': new_correspondence,
        'Status': 'Responded'
    }

    # Update the record in Airtable
    updated_record = airtable_service.update_record(CAMPAIGN_MANAGER_TABLE_NAME, airID, fields_to_update)
    if updated_record:
        print(f"Record {airID} updated successfully.")
    else:
        print(f"Failed to update record {airID}.")