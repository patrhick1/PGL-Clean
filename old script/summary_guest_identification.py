# app/summary_guest_identification.py

import os
import json
import logging
from typing import Optional
import threading
from airtable_service import PodcastService
from anthropic_service import AnthropicService
from openai_service import OpenAIService
from data_processor import generate_prompt, convert_into_json_format

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def process_summary_host_guest(stop_flag: Optional[threading.Event] = None):
    """
    Process podcast episodes to identify summaries, hosts, and guests.
    
    Args:
        stop_flag: Optional threading.Event that signals when to stop processing
    """
    # Initialize services
    airtable_service = PodcastService()
    claude_client = AnthropicService()
    openai_service = OpenAIService()

    try:
        # Check if we should stop before starting
        if stop_flag and stop_flag.is_set():
            logger.info("Stop flag set before starting summary_host_guest processing")
            return

        table_name = "Podcast_Episodes"
        view = "No Summary"

        logger.info(
            f"Fetching records from {table_name} where view is '{view}'...")
        records = airtable_service.get_records_from_view(table_name, view)
        logger.info(f"Found {len(records)} record(s).")

        for record in records:
            # Check stop flag before processing each record
            if stop_flag and stop_flag.is_set():
                logger.info("Stop flag set - stopping summary_host_guest processing")
                return

            record_id = record['id']
            logger.info(f"Processing record ID: {record_id}")

            podcast_records = airtable_service.get_record(
                table_name, record_id)
            fields = podcast_records['fields']

            # Extract fields
            transcription = fields.get('Transcription', '')
            episode_title = fields.get('Episode Title', '')

            podcast_ids = fields.get('Podcast', [])
            if not podcast_ids:
                logger.warning(
                    f"No podcast linked to Podcast Episode record {record_id}"
                )
                continue
            podcast_id = podcast_ids[0]

            # Check if transcription is empty
            if not transcription:
                logger.warning(
                    f"Record {record_id} has no transcription. Skipping...")
                continue

            # Check stop flag before starting analysis
            if stop_flag and stop_flag.is_set():
                logger.info("Stop flag set - stopping before transcript analysis")
                return

            # Step 1: Analyze podcast transcript
            try:
                prompt_path = "prompts/prompt_podcast_transcript_analysis.txt"
                prompt = generate_prompt({'transcription': transcription},
                                         prompt_path)

                logger.info("Sending transcript to Anthropic for analysis...")
                analyze_podcast_transcript = claude_client.create_message(
                    prompt, model='claude-3-5-sonnet-20241022', workflow='summary_guest_identification', podcast_id=podcast_id)
            except Exception as e:
                logger.error(
                    f"Error analyzing podcast transcript for record {record_id}: {e}"
                )
                continue

            # If there's no summary, we skip analyzing further
            if not analyze_podcast_transcript:
                logger.warning(
                    f"Record {record_id} has no summary. Skipping host/guest identification..."
                )
                continue

            # Check stop flag before host/guest analysis
            if stop_flag and stop_flag.is_set():
                logger.info("Stop flag set - stopping before host/guest analysis")
                return

            # Step 2: Analyze the podcast episode (title + summary)
            try:
                prompt_path = "prompts/prompt_podcast_host_guest_analysis.txt"
                prompt = generate_prompt(
                    {
                        'episode_title': episode_title,
                        'summary': transcription
                    }, prompt_path)

                logger.info(
                    "Sending episode info to Anthropic for host/guest analysis..."
                )
                analyze_podcast_episode = claude_client.create_message(
                    prompt, model='claude-3-5-haiku-20241022', workflow='summary_guest_identification', podcast_id=podcast_id)

            except Exception as e:
                logger.error(
                    f"Error figuring out the Host and Guest Name for podcast episode for record {record_id}: {e}"
                )
                continue

            # Check stop flag before JSON conversion
            if stop_flag and stop_flag.is_set():
                logger.info("Stop flag set - stopping before JSON conversion")
                return

            # Step 3: Convert analysis to JSON
            try:
                prompt_path = "prompts/prompt_convert_to_json.txt"
                prompt = generate_prompt(
                    {'analyze_podcast_episode': analyze_podcast_episode},
                    prompt_path)

                logger.info("Requesting JSON conversion from Anthropic...")
                convert_text_to_json = openai_service.transform_text_to_structured_data(
                    prompt, analyze_podcast_episode, data_type='confirmation', workflow='summary_guest_identification_tts', podcast_id=podcast_id)
                logger.info(
                    f"Conversion result for record {record_id}: {convert_text_to_json}"
                )

            except Exception as e:
                logger.error(
                    f"Error converting to JSON for record {record_id}: {e}")
                continue

            # If conversion fails or is empty
            if not convert_text_to_json:
                logger.warning(
                    f"Empty or invalid JSON for record {record_id}. Skipping..."
                )
                continue

            # Check stop flag before validation
            if stop_flag and stop_flag.is_set():
                logger.info("Stop flag set - stopping before validation")
                return

            # Step 4: Extract host, guest, and status from the JSON
            try:
                host = convert_text_to_json.get("Host", '')
                # Check if host is a list
                if isinstance(host, list):
                    # Convert list to a comma-separated string
                    host = ", ".join(map(str, host))

                guest = convert_text_to_json.get('Guest', '')
                # Check if guest is a list
                if isinstance(guest, list):
                    # Convert list to a comma-separated string
                    guest = ", ".join(map(str, guest))

                status = convert_text_to_json.get('status', '')
            except Exception as e:
                logger.error(
                    f"Error reading host/guest from JSON for record {record_id}: {e}"
                )
                continue

            # Step 5: Validate the host/guest labeling
            try:
                prompt_path = "prompts/prompt_validate_host_guest_labeling.txt"
                placeholder = {
                    'Episode Title': episode_title,
                    'summary': analyze_podcast_transcript,
                    'Host': host,
                    'Guest': guest
                }
                prompt = generate_prompt(placeholder, prompt_path)

                logger.info(
                    f"Validating host/guest labeling for record {record_id}..."
                )
                validate_host_guest_labeling = claude_client.create_message(
                    prompt, model='claude-3-5-haiku-20241022', workflow='summary_guest_identification', podcast_id=podcast_id)
                validate_host_guest_labeling = openai_service.transform_text_to_structured_data(
                    prompt,
                    validate_host_guest_labeling,
                    data_type='validation', workflow='summary_guest_identification_tts', podcast_id=podcast_id)

                # If we get a string, try to convert
                if isinstance(validate_host_guest_labeling, str):
                    try:
                        validate_host_guest_labeling = json.loads(
                            validate_host_guest_labeling)
                    except json.JSONDecodeError:
                        logger.error(
                            f"Failed to parse JSON for host/guest validation, record {record_id}."
                        )
                        continue
            except Exception as e:
                logger.error(
                    f"Error validating host/guest labels for record {record_id}: {e}"
                )
                continue

            if not validate_host_guest_labeling:
                logger.warning(
                    f"No validation result returned for record {record_id}. Skipping update..."
                )
                continue

            logger.info(
                f"Validation result for record {record_id}: {validate_host_guest_labeling}"
            )

            # Check stop flag before final update
            if stop_flag and stop_flag.is_set():
                logger.info("Stop flag set - stopping before final update")
                return

            # Step 6: Read the 'correct' value
            correct = validate_host_guest_labeling.get('correct', '')

            # Prepare fields to update in Airtable
            update_fields = {}
            if correct.lower() == 'true':
                if status.lower() == 'both':
                    update_fields = {
                        'Host': host,
                        'Guest': guest,
                        'AI Summary': analyze_podcast_transcript,
                        'Fixed': False,
                        'Guest Confirmed': True,
                        'Flagged Human': False,
                    }
                elif status.lower() == 'host':
                    update_fields = {
                        'Host': host,
                        'AI Summary': analyze_podcast_transcript,
                        'Flagged Human': False,
                    }
                elif status.lower() == 'guest':
                    update_fields = {
                        'Guest': guest,
                        'AI Summary': analyze_podcast_transcript,
                        'Flagged Human': False,
                        'ErrReason': 'Host not confirmed'
                    }
                else:
                    update_fields = {
                        'AI Summary': analyze_podcast_transcript,
                        'Flagged Human': True,
                    }
            else:
                update_fields = {
                    'AI Summary':
                    analyze_podcast_transcript,
                    'Fixed':
                    False,
                    'Guest Confirmed':
                    False,
                    'Flagged Human':
                    True,
                    'ErrReason':
                    ("The junior determined the following and the senior flagged "
                     f"the episode for incorrect labels:\nHost: {host}\nGuest: {guest}"
                     )
                }

            # Step 7: Update Airtable record
            try:
                airtable_service.update_record(table_name, record_id,
                                               update_fields)
                logger.info(
                    f"Successfully updated record {record_id} in Airtable.")
            except Exception as e:
                logger.error(
                    f"Error updating record {record_id} in Airtable: {e}")

    except Exception as e:
        logger.error(f"Error processing records: {e}")
        raise


if __name__ == "__main__":
    logger.info("Starting summary-host-guest identification process...")
    process_summary_host_guest()
    logger.info("Process finished.")
