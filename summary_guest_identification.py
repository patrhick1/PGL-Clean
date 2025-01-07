# app/summary_guest_identification.py

import os
import json
import logging
from airtable_service import PodcastService
from anthropic_service import AnthropicService
from data_processor import generate_prompt, convert_into_json_format

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def process_summary_host_guest():
    # Initialize services
    airtable_service = PodcastService()
    anthropic_service = AnthropicService()

    try:
        table_name = "Podcast_Episodes"
        view = "No Summary"

        logger.info(f"Fetching records from {table_name} where view is '{view}'...")
        records = airtable_service.get_records_from_view(table_name, view)
        logger.info(f"Found {len(records)} record(s).")

        for record in records:
            record_id = record['id']
            logger.info(f"Processing record ID: {record_id}")

            podcast_records = airtable_service.get_record(table_name, record_id)
            fields = podcast_records['fields']

            # Extract fields
            transcription = fields.get('Transcription', '')
            episode_title = fields.get('Episode Title', '')
            summary = fields.get('Summary', '')

            # Check if transcription is empty
            if not transcription:
                logger.warning(f"Record {record_id} has no transcription. Skipping...")
                continue

            # Step 1: Analyze podcast transcript
            try:
                prompt_path = r"app\prompts\prompt_podcast_transcript_analysis.txt"
                prompt = generate_prompt({'transcription': transcription}, prompt_path)
                
                logger.info("Sending transcript to Anthropic for analysis...")
                analyze_podcast_transcript = anthropic_service.create_message(
                    prompt, 
                    model='claude-3-haiku-20240307'
                )
            except Exception as e:
                logger.error(f"Error analyzing podcast transcript for record {record_id}: {e}")
                continue

            # If there's no summary, we skip analyzing further
            if not summary:
                logger.warning(f"Record {record_id} has no summary. Skipping host/guest identification...")
                continue

            # Step 2: Analyze the podcast episode (title + summary)
            try:
                prompt_path = r"app\prompts\prompt_podcast_host_guest_analysis.txt"
                prompt = generate_prompt({'episode_title': episode_title, 'summary': summary}, prompt_path)
                
                logger.info("Sending episode info to Anthropic for host/guest analysis...")
                analyze_podcast_episode = anthropic_service.create_message(
                    prompt, 
                    model='claude-3-haiku-20240307'
                )
            except Exception as e:
                logger.error(f"Error analyzing podcast episode for record {record_id}: {e}")
                continue

            # Step 3: Convert analysis to JSON
            try:
                prompt_path = r"app\prompts\prompt_convert_to_json.txt"
                prompt = generate_prompt({'analyze_podcast_episode': analyze_podcast_episode}, prompt_path)
                
                logger.info("Requesting JSON conversion from Anthropic...")
                convert_text_to_json = anthropic_service.create_message(
                    prompt, 
                    model='claude-3-haiku-20240307'
                )
                convert_text_to_json = convert_into_json_format(convert_text_to_json)
                logger.info(f"Conversion result for record {record_id}: {convert_text_to_json}")
                
            except Exception as e:
                logger.error(f"Error converting to JSON for record {record_id}: {e}")
                continue

            # If conversion fails or is empty
            if not convert_text_to_json:
                logger.warning(f"Empty or invalid JSON for record {record_id}. Skipping...")
                continue

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
                logger.error(f"Error reading host/guest from JSON for record {record_id}: {e}")
                continue

            # Step 5: Validate the host/guest labeling
            try:
                prompt_path = r"app\prompts\prompt_validate_host_guest_labeling.txt"
                placeholder = {
                    'Episode Title': episode_title,
                    'summary': summary,
                    'Host': host,
                    'Guest': guest
                }
                prompt = generate_prompt(placeholder, prompt_path)
                
                logger.info(f"Validating host/guest labeling for record {record_id}...")
                validate_host_guest_labeling = anthropic_service.create_message(
                    prompt, 
                    model='claude-3-haiku-20240307'
                )
                validate_host_guest_labeling = convert_into_json_format(validate_host_guest_labeling)

                # If we get a string, try to convert
                if isinstance(validate_host_guest_labeling, str):
                    try:
                        validate_host_guest_labeling = json.loads(validate_host_guest_labeling)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse JSON for host/guest validation, record {record_id}.")
                        continue
            except Exception as e:
                logger.error(f"Error validating host/guest labels for record {record_id}: {e}")
                continue

            if not validate_host_guest_labeling:
                logger.warning(f"No validation result returned for record {record_id}. Skipping update...")
                continue

            logger.info(f"Validation result for record {record_id}: {validate_host_guest_labeling}")

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
                    'AI Summary': analyze_podcast_transcript,
                    'Fixed': False,
                    'Guest Confirmed': False,
                    'Flagged Human': True,
                    'ErrReason': (
                        "The junior determined the following and the senior flagged "
                        f"the episode for incorrect labels:\nHost: {host}\nGuest: {guest}"
                    )
                }

            # Step 7: Update Airtable record
            try:
                airtable_service.update_record(table_name, record_id, update_fields)
                logger.info(f"Successfully updated record {record_id} in Airtable.")
            except Exception as e:
                logger.error(f"Error updating record {record_id} in Airtable: {e}")

    except Exception as e:
        logger.error(f"Error processing records: {e}")

if __name__ == "__main__":
    logger.info("Starting summary-host-guest identification process...")
    process_summary_host_guest()
    logger.info("Process finished.")
