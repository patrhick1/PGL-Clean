# app/send_pitch.py

from airtable_service import PodcastService
from external_api_service import InstantlyAPI
from dotenv import load_dotenv
import os
import json
import logging

load_dotenv()

def send_pitch_to_instantly():

    logging.info('Starting the Automation to send the pitch to Instantly')

    # Initialize service clients to interact with different APIs
    airtable_client = PodcastService()
    instantly_client = InstantlyAPI()

    # Define the names of tables in Airtable
    CAMPAIGN_MANAGER_TABLE_NAME = 'Campaign Manager'  # Adjust the name if needed
    CAMPAIGNS_TABLE_NAME = 'Campaigns'
    PODCASTS_TABLE_NAME = 'Podcasts'

    # Define view name in the "Campaign Manager" table
    VIEW = 'Pitch Done'

    # Step 1: Fetch all records from the 'Pitch Done' view of the Campaign Manager table
    campaign_manager_records = airtable_client.get_records_from_view(
        CAMPAIGN_MANAGER_TABLE_NAME, 
        VIEW
    )


    # Loop through each record in the Campaign Manager record
    for cm_record in campaign_manager_records:
        try:
            # Get the record ID for reference
            cm_record_id = cm_record['id']
            # Get the fields section from the current record (this is where the data lives)
            cm_fields = cm_record.get('fields', {})

            # Step 2: Get Campaign record
            # Retrieve Campaign ID from Campaign Manager table
            campaign_ids = cm_fields.get('Campaigns', [])
            campaign_id = campaign_ids[0]
            # If there is no campaign linked, skip this record
            if not campaign_ids:
                logging.warning(f"No campaign linked to Campaign Manager record {cm_record_id}")
                continue
            
            # Fetch the actual campaign record from Airtable
            campaign_record = airtable_client.get_record(CAMPAIGNS_TABLE_NAME, campaign_id)

            # Get Podcast record IDs from 'Podcast Name' field in the Campaign Manager record
            podcast_ids = cm_fields.get('Podcast Name', [])
            # If there is no podcast linked, skip this record
            if not podcast_ids:
                logging.warning(f"No podcast linked to Campaign Manager record {cm_record_id}")
                continue

            # Usually only one podcast, so we take the first
            podcast_id = podcast_ids[0]
            # Fetch the podcast record from Airtable
            podcast_record = airtable_client.get_record(PODCASTS_TABLE_NAME, podcast_id)
            
            if not campaign_record or not podcast_record:
                continue

            # Step 4: Prepare data for Instantly API
            api_data = prepare_instantly_api_data(
                campaign_record=campaign_record,
                campaign_manager_record=cm_record,
                podcast_record=podcast_record
            )

            # Step 5: Send data to Instantly API
            response = instantly_client.add_lead(api_data)

            # Step 6: Update Airtable record if response is not 400
            if response.status_code != 400:
                fields = {
                        'Status': 'Instantly'
                    }
                airtable_client.update_record(CAMPAIGN_MANAGER_TABLE_NAME, cm_record_id, fields)
        
        
        except Exception as e:
            # Log an error if something goes wrong while processing this record
            logging.error(f"Error processing Campaign Manager record {cm_record_id}: {str(e)}")
            continue

def prepare_instantly_api_data(campaign_record, campaign_manager_record, podcast_record):
    # Extract fields
    campaign_fields = campaign_record.get('fields', {})
    campaign_manager_fields = campaign_manager_record.get('fields', {})
    podcast_fields = podcast_record.get('fields', {})

    # Prepare the raw data dictionary
    data = {
        "api_key": os.getenv('INSTANTLY_API_KEY'),
        "campaign_id": campaign_fields.get('Instantly', ''),
        "skip_if_in_workspace": False,
        "leads": [
            {
                "email": podcast_fields.get('Email', ''),
                "first_name": podcast_fields.get('Host Name', ''),
                "company_name": podcast_fields.get('Podcast Name', ''),
                "personalization": campaign_manager_fields.get('Pitch Email', ''),
                "custom_variables": {
                    "Client_Name": campaign_fields.get('Name (from Client)', [''])[0],
                    "Subject": campaign_manager_fields.get('Subject Line', ''),
                    "airID": campaign_manager_record['id']
                }
            }
        ]
    }

    # Serialize the data to JSON string if required
    return json.dumps(data)