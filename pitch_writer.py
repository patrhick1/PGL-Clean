"""
Pitch Writer Script

This module completes the final step of creating pitch emails. It fetches data 
from Airtable (e.g., client details, podcast/episode info, angles) and uses an 
LLM (Anthropic Claude) to generate a tailored pitch and subject line. Finally, 
it updates the "Campaign Manager" table with the newly created pitch.

Author: Paschal Okonkwor
Date: 2025-01-06
"""

import logging
from typing import Optional
import threading
from airtable_service import PodcastService
from anthropic_service import AnthropicService
from data_processor import generate_prompt

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def pitch_writer(stop_flag: Optional[threading.Event] = None):
    """
    The main function that handles the creation and update of pitch information 
    in Airtable. It looks in a specific 'View' of the "Campaign Manager" table 
    for records that are at the step of finalizing a pitch. Then, it:
      1. Gathers all relevant data (client bio, summary, selected episode, etc.).
      2. Prompts Claude to write a pitch, along with a subject line.
      3. Updates the "Campaign Manager" record with the pitch and subject.

    Args:
        stop_flag: Optional threading.Event that signals when to stop processing
    """
    logger.info('Starting the Pitch Writer Automation')

    try:
        # Check if we should stop before starting
        if stop_flag and stop_flag.is_set():
            logger.info("Stop flag set before starting pitch_writer")
            return

        # Initialize services to interact with Airtable and Claude
        airtable_client = PodcastService()
        claude_client = AnthropicService()

        # Define table names and view
        CAMPAIGN_MANAGER_TABLE_NAME = 'Campaign Manager'
        CAMPAIGNS_TABLE_NAME = 'Campaigns'
        PODCASTS_TABLE_NAME = 'Podcasts'
        PODCAST_EPISODES_TABLE_NAME = 'Podcast_Episodes'
        EPISODE_AND_ANGLES_VIEW = 'Episode and angles'

        # Fetch all records from the specified view of the "Campaign Manager" table
        campaign_manager_records = airtable_client.get_records_from_view(
            CAMPAIGN_MANAGER_TABLE_NAME, EPISODE_AND_ANGLES_VIEW)
        logger.info(
            f"Fetched {len(campaign_manager_records)} record(s) for pitch writing."
        )

        # Loop over each record
        for cm_record in campaign_manager_records:
            # Check stop flag before processing each record
            if stop_flag and stop_flag.is_set():
                logger.info("Stop flag set - stopping pitch_writer processing")
                return

            cm_record_id = cm_record['id']
            cm_fields = cm_record.get('fields', {})

            try:
                # Check stop flag before fetching associated records
                if stop_flag and stop_flag.is_set():
                    logger.info("Stop flag set - stopping before fetching associated records")
                    return

                # Retrieve the campaign ID
                campaign_ids = cm_fields.get('Campaigns', [])
                if not campaign_ids:
                    logger.warning(
                        f"No campaign linked to Campaign Manager record {cm_record_id}"
                    )
                    continue
                campaign_id = campaign_ids[0]

                # Get the actual campaign record
                campaign_record = airtable_client.get_record(
                    CAMPAIGNS_TABLE_NAME, campaign_id)
                campaign_fields = campaign_record.get('fields', {})

                # Pull the relevant data from the campaign record
                bio = campaign_fields.get('TextBio', '')
                bio_summary = campaign_fields.get('SummaryBio', '')
                client_names = campaign_fields.get('Name (from Client)', [])
                client_name = client_names[
                    0] if client_names else 'No Client Name'

                # Retrieve the podcast ID
                podcast_ids = cm_fields.get('Podcast Name', [])
                if not podcast_ids:
                    logger.warning(
                        f"No podcast linked to Campaign Manager record {cm_record_id}"
                    )
                    continue
                podcast_id = podcast_ids[0]

                # Get the podcast record
                podcast_record = airtable_client.get_record(
                    PODCASTS_TABLE_NAME, podcast_id)
                podcast_fields = podcast_record.get('fields', {})
                podcast_name = podcast_fields.get('Podcast Name', '')
                host_name = podcast_fields.get('Host Name', '')

                # Retrieve the selected podcast episode
                podcast_episode_id = cm_fields.get('Pitch Episode', '')
                if not podcast_episode_id:
                    logger.warning(
                        f"No podcast episode linked to Campaign Manager record {cm_record_id}"
                    )
                    continue

                # Get details about the chosen episode
                podcast_episode_record = airtable_client.get_record(
                    PODCAST_EPISODES_TABLE_NAME, podcast_episode_id)
                podcast_episode_field = podcast_episode_record.get(
                    'fields', {})
                guest_name = podcast_episode_field.get('Guest', '')
                episode_title = podcast_episode_field.get('Episode Title', '')
                episode_summary = podcast_episode_field.get('Summary', '')
                episode_ai_summary = podcast_episode_field.get(
                    'AI Summary', '')

                # Retrieve the pitch topics from the Campaign Manager record
                pitch_topics = cm_fields.get('Pitch Topics', '')

                # Check stop flag before Claude analysis
                if stop_flag and stop_flag.is_set():
                    logger.info("Stop flag set - stopping before Claude analysis")
                    return

                # If there's a guest, we craft the prompt accordingly
                if guest_name:
                    prompt_path = "prompts/pitch_writer_prompt/prompt_pitch_writer.txt"
                    prompt_data = {
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
                    }
                    pitch_prompt = generate_prompt(prompt_data, prompt_path)

                    # Check stop flag before sending to Claude
                    if stop_flag and stop_flag.is_set():
                        logger.info("Stop flag set - stopping before sending to Claude")
                        return

                    # Send prompt to Claude
                    write_pitch = claude_client.create_message(
                        pitch_prompt, model='claude-3-5-sonnet-20241022', workflow='pitch_writer', podcast_id=podcast_id)

                    # Check stop flag before updating Airtable
                    if stop_flag and stop_flag.is_set():
                        logger.info("Stop flag set - stopping before Airtable update")
                        return

                    # Build a subject line
                    subject = f"Great episode with {guest_name}"

                    # Update Airtable
                    update_fields = {
                        'Status': 'Pitch Done',
                        'Pitch Email': write_pitch,
                        'Subject Line': subject
                    }
                    airtable_client.update_record(CAMPAIGN_MANAGER_TABLE_NAME,
                                                  cm_record_id, update_fields)

                else:
                    # Check stop flag before no-guest processing
                    if stop_flag and stop_flag.is_set():
                        logger.info("Stop flag set - stopping before no-guest processing")
                        return

                    # If there's no guest, we use a slightly different prompt
                    prompt_path = "prompts/pitch_writer_prompt/prompt_pitch_writer.txt"
                    prompt_data = {
                        'Podcast Name': podcast_name,
                        'Host Name': host_name,
                        'Episode Title': episode_title,
                        'Summary': episode_summary,
                        'AI Summary': episode_ai_summary,
                        'SummaryBio': bio_summary,
                        'Pitch Topics': pitch_topics,
                        'Name (from Client)': client_name,
                        'TextBio': bio,
                    }
                    pitch_prompt = generate_prompt(prompt_data, prompt_path)
                    write_pitch = claude_client.create_message(
                        pitch_prompt, model='claude-3-5-sonnet-20241022', workflow='pitch_writer', podcast_id=podcast_id)

                    # Get a subject line from Claude
                    subject_prompt_path = "prompts/pitch_writer_prompt/prompt_write_subject_line.txt"
                    subject_prompt_data = {
                        'Summary': episode_summary,
                        'AI Summary': episode_ai_summary,
                    }
                    subject_prompt = generate_prompt(subject_prompt_data,
                                                     subject_prompt_path)
                    subject = claude_client.create_message(subject_prompt, workflow='pitch_writer', podcast_id=podcast_id)

                    # Check stop flag before final update
                    if stop_flag and stop_flag.is_set():
                        logger.info("Stop flag set - stopping before final update")
                        return

                    # Update the record
                    update_fields = {
                        'Status': 'Pitch Done',
                        'Pitch Email': write_pitch,
                        'Subject Line': subject
                    }
                    airtable_client.update_record(CAMPAIGN_MANAGER_TABLE_NAME,
                                                  cm_record_id, update_fields)

                logger.info(
                    f"Campaign Manager record {cm_record_id} updated with new pitch."
                )

            except Exception as e:
                logger.error(
                    f"Error processing Campaign Manager record {cm_record_id}: {str(e)}"
                )
                continue

    except Exception as e:
        logger.error(f"Error in pitch_writer function: {str(e)}")
        raise
