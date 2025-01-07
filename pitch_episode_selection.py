#app/pitch_episode_selection.py

import logging
from airtable_service import PodcastService
from anthropic_service import AnthropicService
from google_docs_service import GoogleDocsService
from data_processor import generate_prompt, convert_into_json_format


def pitch_episode_selection():
    logging.info('Starting Pitch and Angles selection Automation')

    # Initialize the services
    airtable_client = PodcastService()
    claude_client = AnthropicService()
    google_docs_client = GoogleDocsService()

    # Define the Airtable table names
    CAMPAIGN_MANAGER_TABLE_NAME = 'Campaign Manager'  # Adjust as needed
    CAMPAIGNS_TABLE_NAME = 'Campaigns'
    PODCASTS_TABLE_NAME = 'Podcasts'
    PODCAST_EPISODES_TABLE_NAME = 'Podcast_Episodes'

    #Define views in podcast table
    OUTREACH_READY_VIEW = 'Fit'

    # Step 1: Fetch records from Campaign Manager from the OR View"

    campaign_manager_records = airtable_client.get_records_from_view(CAMPAIGN_MANAGER_TABLE_NAME, OUTREACH_READY_VIEW)
    # Process each record
    print("Processing each record in campaign manager")
    for cm_record in campaign_manager_records:
        try:
            cm_record_id = cm_record['id']
            cm_fields = cm_record.get('fields', {})

            # Get Campaign record linked in 'Campaigns' field
            campaign_ids = cm_fields.get('Campaigns', [])
            if not campaign_ids:
                print(f"No campaign linked to Campaign Manager record {cm_record_id}")
                logging.warning(f"No campaign linked to Campaign Manager record {cm_record_id}")
                continue
            campaign_id = campaign_ids[0]  # Assuming single campaign
            campaign_record = airtable_client.get_record(CAMPAIGNS_TABLE_NAME, campaign_id)
            campaign_fields = campaign_record.get('fields', {})
            bio = campaign_fields.get('TextBio', '')
            angles = campaign_fields.get('TextAngles', '')
            client_names = campaign_fields.get('Name (from Client)', [])
            client_name = client_names[0]

            
            # Get Podcast record linked in 'Podcast Name' field
            podcast_ids = cm_fields.get('Podcast Name', [])
            if not podcast_ids:
                logging.warning(f"No podcast linked to Campaign Manager record {cm_record_id}")
                continue
            podcast_id = podcast_ids[0]

            podcast_record = airtable_client.get_record(PODCASTS_TABLE_NAME, podcast_id)
            podcast_fields = podcast_record.get('fields', {})
            podcast_name = podcast_fields.get('Podcast Name', '')
            podcast_episode_info = podcast_fields.get('PodcastEpisodeInfo', '')

            #get the content from the podcast episode document
            podcast_episode_content = google_docs_client.get_document_content(podcast_episode_info)

            # Now, prepare prompt to send to Claude to generate the relevant episodes ids
            prompt_path = r"app\prompts\pitch_episodes_angles_selection_prompts\prompt_claude_get_episode_id.txt"
            prompt = generate_prompt({'Name (from Client)':client_name, 
                                      'Podcast Name': podcast_name,
                                      'TextBio':bio,
                                      'TextAngles':angles,
                                      'text':podcast_episode_content}, prompt_path)
            # Send prompt to Claude
            get_episode_ids = claude_client.create_message(prompt)
            
            
            # Now, prepare prompt to send to Claude to generate the relevant episodes ids
            prompt_path = r"app\prompts\prompt_extract_json.txt"
            prompt = generate_prompt({'textResponse':get_episode_ids}, prompt_path)
            # Send prompt to Claude to get the episode id
            extract_episode_id = claude_client.create_message(prompt)
            extract_episode_id = convert_into_json_format(extract_episode_id)
            extract_episode_id = extract_episode_id.get('ID')


            #use the episode id to retrieve the podcast episode record
            podcast_episode_record = airtable_client.get_record(PODCAST_EPISODES_TABLE_NAME, extract_episode_id)
            podcast_episode_field = podcast_episode_record.get('fields', {})
            episode_title = podcast_episode_field.get('Episode Title','')
            episode_summary = podcast_episode_field.get('Summary','')
            episode_ai_summary = podcast_episode_field.get('AI Summary','')


            # prepare prompt to send to Claude to generate the pitch idea
            prompt_path = r"app\prompts\pitch_episodes_angles_selection_prompts\prompt_write_pitch.txt"
            prompt = generate_prompt({'Name (from Client)':client_name, 
                                      'Podcast Name': podcast_name,
                                      'Episode Title': episode_title,
                                      'Summary':episode_summary,
                                      'AI Summary':episode_ai_summary,
                                      'TextAngles':angles,}, prompt_path)
            
            # Send prompt to Claude to write the pitch
            write_pitch = claude_client.create_message(prompt)

            # Now, prepare prompt to send to Claude to extract only the json format of the pitch
            prompt_path = r"app\prompts\pitch_episodes_angles_selection_prompts\prompt_extract_json.txt"
            prompt = generate_prompt({'textResponse':write_pitch}, prompt_path)
            # Send prompt to Claude to get the episode id
            pitch_ideas = claude_client.create_message(prompt)
            pitch_ideas = convert_into_json_format(pitch_ideas)

            # Update 'Status', 'Pitch Episode' and Topic field in Campaign Manager record
            update_fields = {
                'Status': 'Episode and angles selected',
                'Pitch Episode': extract_episode_id,
                'Pitch Topics': f"""
1. {pitch_ideas.get('Topic 1')} : {pitch_ideas.get('Description 1')}
2. {pitch_ideas.get('Topic 2')} : {pitch_ideas.get('Description 2')}
3.  {pitch_ideas.get('Topic 3')}: {pitch_ideas.get('Description 3')}
                """
            }

            #Update Campaign manager table
            airtable_client.update_record(CAMPAIGN_MANAGER_TABLE_NAME, cm_record_id, update_fields)


        except Exception as e:
            logging.error(f"Error processing Campaign Manager record {cm_record_id}: {str(e)}")
            continue

if __name__ == "__main__":
    pitch_episode_selection()

