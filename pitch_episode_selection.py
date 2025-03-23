"""
Pitch Episode Selection Script

This module selects the best podcast episode and angle for pitching a client 
to a specific podcast. It uses data from Airtable (Campaign Manager, Campaigns,
Podcasts, etc.), reads summarized episode information from Google Docs, and
prompts Claude to determine which episodes are best for pitching.

Author: Paschal Okonkwor
Date: 2025-01-06
"""

import logging
from typing import Optional
import threading
from airtable_service import PodcastService
from anthropic_service import AnthropicService
from openai_service import OpenAIService
from google_docs_service import GoogleDocsService
from data_processor import generate_prompt, convert_into_json_format

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def pitch_episode_selection(stop_flag: Optional[threading.Event] = None):
    """
    This function automates selecting the right podcast episode for pitching.
    It fetches records from Airtable's "Campaign Manager" (in the "Fit" view),
    then uses Claude to:
      1. Identify which episode is best for the client.
      2. Generate a pitch with relevant angles.
    Finally, it updates the record in Airtable with the chosen episode and topics.

    Args:
        stop_flag: Optional threading.Event that signals when to stop processing
    """
    logger.info('Starting Pitch and Angles Selection Automation')

    try:
        # Check if we should stop before starting
        if stop_flag and stop_flag.is_set():
            logger.info("Stop flag set before starting pitch_episode_selection")
            return

        # Initialize services
        airtable_client = PodcastService()
        claude_client = AnthropicService()
        openai_client = OpenAIService()
        google_docs_client = GoogleDocsService()

        # Define table names
        CAMPAIGN_MANAGER_TABLE_NAME = 'Campaign Manager'
        CAMPAIGNS_TABLE_NAME = 'Campaigns'
        PODCASTS_TABLE_NAME = 'Podcasts'
        PODCAST_EPISODES_TABLE_NAME = 'Podcast_Episodes'

        # Define the view name in the Campaign Manager table
        OUTREACH_READY_VIEW = 'Fit'

        # Step 1: Fetch records from the "Fit" view
        campaign_manager_records = airtable_client.get_records_from_view(
            CAMPAIGN_MANAGER_TABLE_NAME, OUTREACH_READY_VIEW)
        logger.info(
            f"Fetched {len(campaign_manager_records)} record(s) from the 'Fit' view."
        )

        # Process each record
        for cm_record in campaign_manager_records:
            # Check stop flag before processing each record
            if stop_flag and stop_flag.is_set():
                logger.info("Stop flag set - stopping pitch_episode_selection processing")
                return

            cm_record_id = cm_record['id']
            cm_fields = cm_record.get('fields', {})

            try:
                # Check stop flag before fetching associated records
                if stop_flag and stop_flag.is_set():
                    logger.info("Stop flag set - stopping before fetching associated records")
                    return

                # Fetch associated Campaign record
                campaign_ids = cm_fields.get('Campaigns', [])
                if not campaign_ids:
                    logger.warning(
                        f"No campaign linked to Campaign Manager record {cm_record_id}"
                    )
                    continue
                campaign_id = campaign_ids[0]
                campaign_record = airtable_client.get_record(
                    CAMPAIGNS_TABLE_NAME, campaign_id)
                campaign_fields = campaign_record.get('fields', {})
                bio = campaign_fields.get('TextBio', '')
                angles = campaign_fields.get('TextAngles', '')
                client_names = campaign_fields.get('Name (from Client)', [])
                client_name = client_names[
                    0] if client_names else 'Unknown Client'

                # Fetch associated Podcast record
                podcast_ids = cm_fields.get('Podcast Name', [])
                if not podcast_ids:
                    logger.warning(
                        f"No podcast linked to Campaign Manager record {cm_record_id}"
                    )
                    continue
                podcast_id = podcast_ids[0]
                podcast_record = airtable_client.get_record(
                    PODCASTS_TABLE_NAME, podcast_id)
                podcast_fields = podcast_record.get('fields', {})
                podcast_name = podcast_fields.get('Podcast Name', '')
                podcast_episode_info = podcast_fields.get(
                    'PodcastEpisodeInfo', '')

                # Get content from the "Podcast Episode Info" Google Doc
                podcast_episode_content = google_docs_client.get_document_content(
                    podcast_episode_info)

                # Check stop flag before Claude analysis
                if stop_flag and stop_flag.is_set():
                    logger.info("Stop flag set - stopping before Claude analysis")
                    return

                # 1. Prompt Claude to identify the best episode IDs
                prompt_path = "prompts/pitch_episodes_angles_selection_prompts/prompt_claude_get_episode_id.txt"
                prompt = generate_prompt(
                    {
                        'Name (from Client)': client_name,
                        'Podcast Name': podcast_name,
                        'TextBio': bio,
                        'TextAngles': angles,
                        'text': podcast_episode_content
                    }, prompt_path)
                get_episode_ids = claude_client.create_message(
                    prompt, model='claude-3-5-sonnet-20241022', workflow='pitch_episode_selection', podcast_id=podcast_id)
                print(get_episode_ids)
                logger.info(
                    f"Successfully fetched episode IDs from document {podcast_episode_info}"
                )

                # Check stop flag before extracting episode ID
                if stop_flag and stop_flag.is_set():
                    logger.info("Stop flag set - stopping before episode ID extraction")
                    return

                # 2. Extract the episode ID from Claude's response
                prompt_path = "prompts/prompt_extract_json.txt"
                json_prompt = generate_prompt(
                    {'textResponse': get_episode_ids}, prompt_path)
                episode_id_extract = openai_client.transform_text_to_structured_data(
                    json_prompt, get_episode_ids, 'episode_ID', workflow='pitch_episode_selection_tts', podcast_id=podcast_id)
                chosen_episode_id = episode_id_extract.get('ID')

                # Retrieve the chosen Podcast Episode record
                podcast_episode_record = airtable_client.get_record(
                    PODCAST_EPISODES_TABLE_NAME, chosen_episode_id)
                #print(podcast_episode_record)
                podcast_episode_field = podcast_episode_record.get(
                    'fields', {})
                episode_title = podcast_episode_field.get('Episode Title', '')
                episode_summary = podcast_episode_field.get('Summary', '')
                episode_ai_summary = podcast_episode_field.get(
                    'AI Summary', '')

                # 3. Prompt Claude to write the pitch based on chosen episode
                pitch_prompt_path = "prompts/pitch_episodes_angles_selection_prompts/prompt_write_pitch.txt"
                pitch_prompt = generate_prompt(
                    {
                        'Name (from Client)': client_name,
                        'Podcast Name': podcast_name,
                        'Episode Title': episode_title,
                        'Summary': episode_summary,
                        'AI Summary': episode_ai_summary,
                        'TextAngles': angles,
                    }, pitch_prompt_path)
                write_pitch = claude_client.create_message(
                    pitch_prompt, model='claude-3-5-sonnet-20241022', workflow='pitch_episode_selection', podcast_id=podcast_id)

                # 4. Extract the JSON version of the pitch from Claude
                extract_prompt_path = "prompts/pitch_episodes_angles_selection_prompts/prompt_extract_json.txt"
                extract_prompt = generate_prompt({'textResponse': write_pitch},
                                                 extract_prompt_path)
                pitch_ideas = openai_client.transform_text_to_structured_data(
                    extract_prompt, write_pitch, 'topic_descriptions', workflow='pitch_episode_selection_tts', podcast_id=podcast_id)

                # Build the final string for the pitch topics
                update_fields = {
                    'Status':
                    'Episode and angles selected',
                    'Pitch Episode':
                    chosen_episode_id,
                    'Pitch Topics':
                    f"""
1. {pitch_ideas.get('topic_1')} : {pitch_ideas.get('description_1')}
2. {pitch_ideas.get('topic_2')} : {pitch_ideas.get('description_2')}
3. {pitch_ideas.get('topic_3')}: {pitch_ideas.get('description_3')}
                    """
                }

                # Check stop flag before updating Airtable
                if stop_flag and stop_flag.is_set():
                    logger.info("Stop flag set - stopping before Airtable update")
                    return

                # 5. Update Campaign Manager with the chosen episode and pitch
                airtable_client.update_record(CAMPAIGN_MANAGER_TABLE_NAME,
                                              cm_record_id, update_fields)
                logger.info(
                    f"Record {cm_record_id} updated with chosen episode {chosen_episode_id} and new pitch topics."
                )

            except Exception as e:
                logger.error(
                    f"Error processing Campaign Manager record {cm_record_id}: {str(e)}"
                )
                continue

    except Exception as e:
        logger.error(f"Error in pitch_episode_selection function: {str(e)}")
        raise
