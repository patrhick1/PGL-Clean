# app/send_pitch.py

from .airtable_service import PodcastService
from .external_api_service import InstantlyAPI
from dotenv import load_dotenv
import os
import json
import logging
import sys # Import sys to exit script

load_dotenv()

def send_pitch_to_instantly(stop_flag=None):

    # Initialize service clients to interact with different APIs
    airtable_client = PodcastService()
    instantly_client = InstantlyAPI()

    # ===== API Key Check =====
    logging.info("Checking Instantly API key...")
    test_response = instantly_client.list_campaigns()

    if test_response is None:
        logging.critical("Failed to make test request to Instantly API. Check network or InstantlyAPI class.")
        sys.exit(1) # Exit script if request failed
    
    if test_response.status_code == 200:
        logging.info("Instantly API key seems valid. Proceeding...")
    elif test_response.status_code == 401:
        logging.critical("Instantly API key is invalid or missing (401 Unauthorized). Please check .env file.")
        sys.exit(1) # Exit script
    elif test_response.status_code == 403:
        logging.critical("Instantly API key does not have permission to list campaigns (403 Forbidden). Check API key permissions in Instantly.")
        sys.exit(1) # Exit script
    else:
        # Log other unexpected errors but maybe proceed cautiously or exit depending on policy
        logging.error(f"Unexpected status code {test_response.status_code} from Instantly API during key check. Response: {test_response.text}")
        # Decide whether to exit based on the error
        sys.exit(1) # Exit on other errors for now

    logging.info('Starting the Automation to send the pitch to Instantly')

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
    logging.info(f"Found {len(campaign_manager_records)} records in '{VIEW}' view.")

    # Loop through each record in the Campaign Manager record
    for cm_record in campaign_manager_records:
        cm_record_id = cm_record['id']
        logging.info(f"Processing Campaign Manager record ID: {cm_record_id}")
        try:
            # Check if the stop flag is set before processing each record
            if stop_flag and stop_flag.is_set():
                logging.info("Stop flag set, halting send_pitch_to_instantly.")
                break

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

            podcast_fields = podcast_record.get('fields', {})
            raw_emails_str = podcast_fields.get('Email')

            if not raw_emails_str:
                logging.warning(f"No email found for Podcast {podcast_id} linked to CM record {cm_record_id}")
                continue
            
            # Split emails by comma, strip whitespace, and filter out empty strings
            email_list = [email.strip() for email in raw_emails_str.split(',') if email.strip()]
            
            any_success = False # Track if any email was sent successfully

            for current_email in email_list:
                # Step 4: Prepare data for Instantly API for the current email
                api_data = prepare_instantly_api_data(
                    campaign_record=campaign_record,
                    campaign_manager_record=cm_record,
                    podcast_record=podcast_record,
                    email=current_email # Pass the current email
                )
                logging.debug(f"Data prepared for Instantly API for {cm_record_id} with email {current_email}: {api_data}")

                # Step 5: Send data to Instantly API
                logging.info(f"Sending data to Instantly for {cm_record_id} with email: {current_email}...")
                response = instantly_client.add_lead_v2(api_data)
                #logging.info(f"Instantly API response for {cm_record_id} (Email: {current_email}) - Status: {response.status_code}, Body: {response.text}")
                logging.info(f"Instantly API response status for email {current_email}: {response.status_code}")

                # Step 6: Check response for the current email
                if response.status_code == 200:
                    any_success = True # Mark success if at least one email works
                    logging.info(f"Successfully sent to Instantly for email: {current_email}")
                elif response.status_code == 400:
                    logging.warning(f"Instantly API returned status 400 for {cm_record_id} with email {current_email}. Might be an invalid email or duplicate.")
                else:
                    logging.error(f"Instantly API returned status {response.status_code} for {cm_record_id} with email {current_email}.")

            # Step 7: Update Airtable record only if at least one email was sent successfully
            if any_success:
                fields = {
                        'Status': 'Instantly'
                    }
                logging.info(f"Updating Airtable record {cm_record_id} status to 'Instantly' as at least one email succeeded.")
                airtable_client.update_record(CAMPAIGN_MANAGER_TABLE_NAME, cm_record_id, fields)
            else:
                logging.warning(f"Skipped Airtable update for {cm_record_id} as no emails were successfully sent to Instantly.")
        
        
        except Exception as e:
            # Log an error if something goes wrong while processing this record
            logging.error(f"Error processing Campaign Manager record {cm_record_id}: {str(e)}", exc_info=True)
            continue

def prepare_instantly_api_data(campaign_record, campaign_manager_record, podcast_record, email):
    # Extract fields
    campaign_fields = campaign_record.get('fields', {})
    campaign_manager_fields = campaign_manager_record.get('fields', {})
    podcast_fields = podcast_record.get('fields', {})

    # Prepare the raw data dictionary
    data = {
        "campaign": campaign_fields.get('Instantly', ''),
        "skip_if_in_workspace": True,
        "skip_if_in_campaign": True,
        "email": email,
        "first_name": podcast_fields.get('Host Name', ''),
        "company_name": podcast_fields.get('Podcast Name', ''),
        "personalization": campaign_manager_fields.get('Pitch Email', ''),
        "custom_variables": {
            "Client_Name": campaign_fields.get('Name (from Client)', [''])[0],
            "Subject": campaign_manager_fields.get('Subject Line', ''),
            "airID": campaign_manager_record['id']
        }
    }

    # Serialize the data to JSON string if required
    return json.dumps(data)

if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info("Script execution started directly.")
    try:
        logging.info("Calling send_pitch_to_instantly function...")
        send_pitch_to_instantly()
        logging.info("send_pitch_to_instantly function completed.")
    except Exception as e:
        logging.error(f"An error occurred during script execution: {str(e)}", exc_info=True)
    logging.info("Script execution finished.")