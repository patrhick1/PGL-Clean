# app/pitch_writer.py

import logging
from airtable_service import PodcastService
from anthropic_service import AnthropicService
from google_docs_service import GoogleDocsService
from data_processor import generate_prompt

def pitch_writer():
    """
    Main function that handles creating and updating pitch information in Airtable.
    This includes collecting data from the "Campaign Manager" records and then
    using Anthropic's Claude to generate pitches and subject lines.
    """
    logging.info('Starting the Pitch Writer Automation')

    # Initialize service clients to interact with different APIs
    airtable_client = PodcastService()
    claude_client = AnthropicService()

    # Define the names of tables in Airtable
    CAMPAIGN_MANAGER_TABLE_NAME = 'Campaign Manager'  # Adjust the name if needed
    CAMPAIGNS_TABLE_NAME = 'Campaigns'
    PODCASTS_TABLE_NAME = 'Podcasts'
    PODCAST_EPISODES_TABLE_NAME = 'Podcast_Episodes'

    # Define view name in the "Campaign Manager" table
    EPISODE_AND_ANGLES = 'Episode and angles'

    # Step 1: Fetch all records from the 'Episode and angles' view of the Campaign Manager table
    campaign_manager_records = airtable_client.get_records_from_view(
        CAMPAIGN_MANAGER_TABLE_NAME, 
        EPISODE_AND_ANGLES
    )

    # Loop through each record in the Campaign Manager table
    for cm_record in campaign_manager_records:
        try:
            # Get the record ID for reference
            cm_record_id = cm_record['id']
            # Get the fields section from the current record (this is where the data lives)
            cm_fields = cm_record.get('fields', {})

            # Retrieve related Campaign record IDs from 'Campaigns' field
            campaign_ids = cm_fields.get('Campaigns', [])
            # Retrieve pitch topics from the current record
            pitch_topics = cm_fields.get('Pitch Topics','')

            # If there is no campaign linked, skip this record
            if not campaign_ids:
                logging.warning(f"No campaign linked to Campaign Manager record {cm_record_id}")
                continue

            # Since there's usually only one campaign, just take the first one
            campaign_id = campaign_ids[0]
            # Fetch the actual campaign record from Airtable
            campaign_record = airtable_client.get_record(CAMPAIGNS_TABLE_NAME, campaign_id)
            # Extract the fields from the campaign record
            campaign_fields = campaign_record.get('fields', {})

            # Get relevant info from the campaign record
            bio = campaign_fields.get('TextBio', '')
            bio_summary = campaign_fields.get('SummaryBio', '')
            client_names = campaign_fields.get('Name (from Client)', [])
            client_name = client_names[0] if client_names else 'No Client Name'

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
            # Extract fields from the podcast record
            podcast_fields = podcast_record.get('fields', {})
            podcast_name = podcast_fields.get('Podcast Name', '')
            host_name = podcast_fields.get('Host Name', '')

            # Get the Podcast Episode ID from 'Pitch Episode' field in the Campaign Manager record
            podcast_episode_id = cm_fields.get('Pitch Episode', '')
            # If there is no podcast episode linked, skip
            if not podcast_episode_id:
                logging.warning(f"No podcast episode linked to Campaign Manager record {cm_record_id}")
                continue

            # Fetch the podcast episode record from Airtable
            podcast_episode_record = airtable_client.get_record(PODCAST_EPISODES_TABLE_NAME, podcast_episode_id)
            # Extract fields from the podcast episode record
            podcast_episode_field = podcast_episode_record.get('fields', {})
            guest_name = podcast_episode_field.get('Guest', '')
            episode_title = podcast_episode_field.get('Episode Title', '')
            episode_summary = podcast_episode_field.get('Summary', '')
            episode_ai_summary = podcast_episode_field.get('AI Summary', '')

            # If there's a guest, we'll use a different prompt than if there's no guest
            if guest_name:
                # Prepare the prompt to send to Claude to generate the pitch (for episodes with a guest)
                prompt_path = r"app\prompts\pitch_writer_prompt\prompt_pitch_writer.txt"
                
                # We call a function to generate the prompt text by filling it with our dynamic data
                prompt = generate_prompt({
                    'Podcast Name': podcast_name,
                    'Host Name': host_name,
                    'Guest': guest_name,
                    'Episode Title': episode_title,
                    'Summary': episode_summary,
                    'AI Summary': episode_ai_summary,
                    'SummaryBio': bio_summary,
                    'Pitch Topics': pitch_topics,
                    'Name (from Client)': client_name,
                    'TextBio': bio,
                }, prompt_path)
                
                # Send the prompt to Claude to create the pitch
                write_pitch = claude_client.create_message(prompt)

                # Build a subject line for our pitch
                subject = f"Great episode with {guest_name}"

                # Prepare fields to update in the Campaign Manager table
                update_fields = {
                    'Status': 'Pitch Done',
                    'Pitch Email': write_pitch,
                    'Subject Line': subject
                }

                # Finally, update the Campaign Manager record with the new pitch and subject
                airtable_client.update_record(CAMPAIGN_MANAGER_TABLE_NAME, cm_record_id, update_fields)

            else:
                # If there is no guest, we use a different prompt
                prompt_path = r"app\prompts\pitch_writer_prompt\prompt_pitch_writer.txt"
                
                # Generate the pitch prompt for episodes without a guest
                prompt = generate_prompt({
                    'Podcast Name': podcast_name,
                    'Host Name': host_name,
                    'Episode Title': episode_title,
                    'Summary': episode_summary,
                    'AI Summary': episode_ai_summary,
                    'SummaryBio': bio_summary,
                    'Pitch Topics': pitch_topics,
                    'Name (from Client)': client_name,
                    'TextBio': bio,
                }, prompt_path)
                
                # Send this prompt to Claude to create the pitch
                write_pitch = claude_client.create_message(prompt)

                # Use another prompt to get a suggested subject line from Claude
                subject_prompt_path = r"app\prompts\pitch_writer_prompt\prompt_write_subject_line.txt"
                subject_prompt = generate_prompt({
                    'Summary': episode_summary,
                    'AI Summary': episode_ai_summary,
                }, subject_prompt_path)
                
                subject = claude_client.create_message(subject_prompt)

                # Prepare fields to update in the Campaign Manager table
                update_fields = {
                    'Status': 'Pitch Done',
                    'Pitch Email': write_pitch,
                    'Subject Line': subject
                }

                # Update the record in the Campaign Manager table
                airtable_client.update_record(CAMPAIGN_MANAGER_TABLE_NAME, cm_record_id, update_fields)
                print(f"Record ID: {cm_record_id} has been updated")

        except Exception as e:
            # Log an error if something goes wrong while processing this record
            logging.error(f"Error processing Campaign Manager record {cm_record_id}: {str(e)}")
            continue

if __name__ == "__main__":
    pitch_writer()

