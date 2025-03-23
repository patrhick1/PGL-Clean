"""
Determine Fit Script

This module checks records in your "Campaign Manager" table and decides
whether a podcast is a good fit for a client. It pulls data from Airtable,
fetches existing Google Docs content, and leverages an LLM (Anthropic Claude)
to evaluate fit. The status of each record is then updated in Airtable.

Author: Paschal Okonkwor
Date: 2025-01-06
"""

import logging
import os
import time
import re
import threading
from typing import Optional

from dotenv import load_dotenv
from airtable_service import PodcastService
from anthropic_service import AnthropicService
from openai_service import OpenAIService
from google_docs_service import GoogleDocsService
from data_processor import generate_prompt, convert_into_json_format

# Load environment variables (e.g., for Google Docs or Airtable tokens)
load_dotenv()

# Configure logging to show messages in the terminal
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def sanitize_filename(name):
    # Remove emojis and other non-ASCII characters
    name = re.sub(r'[^\w\s-]', '', name, flags=re.UNICODE)
    # Replace spaces and other unsafe characters with underscores
    name = re.sub(r'[\s/\\]', '_', name)
    # Trim leading/trailing underscores
    return name.strip('_')


def determine_fit(stop_flag: Optional[threading.Event] = None):
    """
    Main function to check which podcasts are a good fit for the client. 
    It fetches records from an Airtable "Campaign Manager" view, processes each
    record with help from Anthropics' Claude, and updates the "Status" field.

    Args:
        stop_flag: Optional threading.Event that signals when to stop processing

    Steps:
    1. Fetch relevant records from the "Campaign Manager" table using a specific view.
    2. Retrieve campaign and podcast info from Airtable.
    3. Collect episode details in Google Docs if needed.
    4. Send the content and client info to Claude for analysis.
    5. Update the record's status in Airtable based on Claude's response.
    """
    logger.info('Starting Determine Fit Automation')

    try:
        # STEP 0: Initialize service clients
        logger.info("Initializing Airtable, Claude, and Google Docs services.")
        airtable_client = PodcastService()
        claude_client = AnthropicService()
        openai_service = OpenAIService()
        google_docs_client = GoogleDocsService()

        # Check if we should stop before starting
        if stop_flag and stop_flag.is_set():
            logger.info("Stop flag set before starting determine_fit")
            return

        # Folder in Google Drive where your docs will be stored
        PODCAST_INFO_FOLDER_ID = os.getenv('GOOGLE_PODCAST_INFO_FOLDER_ID')
        logger.info(
            f"PODCAST_INFO_FOLDER_ID is set to: {PODCAST_INFO_FOLDER_ID}")

        # Define table and view names
        CAMPAIGN_MANAGER_TABLE_NAME = 'Campaign Manager'
        CAMPAIGNS_TABLE_NAME = 'Campaigns'
        PODCASTS_TABLE_NAME = 'Podcasts'
        PODCAST_EPISODES_TABLE_NAME = 'Podcast_Episodes'
        OUTREACH_READY_VIEW = 'OR'

        # STEP 1: Fetch all records from the "OR" view in the Campaign Manager table
        logger.info(
            "Fetching records from the 'OR' view of the Campaign Manager table."
        )
        campaign_manager_records = airtable_client.get_records_from_view(
            CAMPAIGN_MANAGER_TABLE_NAME, OUTREACH_READY_VIEW)
        logger.info(
            f"Found {len(campaign_manager_records)} record(s) in the 'OR' view."
        )

        # Process each record found
        for cm_record in campaign_manager_records:
            # Check stop flag before processing each record
            if stop_flag and stop_flag.is_set():
                logger.info("Stop flag set - stopping determine_fit processing")
                return

            cm_record_id = cm_record['id']
            cm_fields = cm_record.get('fields', {})
            logger.info(
                f"Processing Campaign Manager record with ID: {cm_record_id}")

            try:
                # STEP 2: Retrieve associated Campaign record
                logger.info("Retrieving associated Campaign record.")
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

                logger.info(
                    f"Campaign record retrieved: {campaign_id} | Bio length: {len(bio)}, Angles length: {len(angles)}"
                )

                # STEP 3: Retrieve associated Podcast record
                logger.info("Retrieving associated Podcast record.")
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
                sanitized_podcast_name = sanitize_filename(podcast_name)
                logger.info(
                    f"Podcast record retrieved: {podcast_id} | Podcast Name: {podcast_name}"
                )

                # STEP 4: Create or find the "Podcast Info" Google Doc
                logger.info(
                    f"Checking if a Google Doc named '{podcast_name} - Info' already exists."
                )
                google_doc_name = f"{sanitized_podcast_name} - Info"
                doc_search_result = google_docs_client.check_file_exists_in_folder(
                    google_doc_name)

                if doc_search_result[0] is False:
                    # Document doesn't exist
                    logger.info(
                        f"No existing document found. Creating new document '{google_doc_name}'."
                    )
                    google_doc_id = google_docs_client.create_document_without_content(
                        google_doc_name, PODCAST_INFO_FOLDER_ID)
                    logger.info(
                        f"New document created with ID: {google_doc_id}")
                    # Save doc ID in Airtable
                    update_fields = {'PodcastEpisodeInfo': google_doc_id}
                    airtable_client.update_record(PODCASTS_TABLE_NAME,
                                                  podcast_id, update_fields)
                    logger.info(
                        f"New doc created: {google_doc_id} and stored in Airtable under 'PodcastEpisodeInfo'."
                    )

                    # Collect and append each episode's data to the new doc
                    episode_ids = podcast_fields.get('Podcast Episodes', [])
                    if not episode_ids:
                        logger.warning(
                            f"No episodes linked to Podcast record {podcast_id}"
                        )
                        continue

                    episode_summaries = ''
                    logger.info(
                        f"Found {len(episode_ids)} episode(s) for Podcast record {podcast_id}. Appending them to the new doc."
                    )
                    for episode_id in episode_ids:
                        # Check stop flag before each episode
                        if stop_flag and stop_flag.is_set():
                            logger.info("Stop flag set - stopping episode processing")
                            return

                        episode_record = airtable_client.get_record(
                            PODCAST_EPISODES_TABLE_NAME, episode_id)
                        episode_fields = episode_record.get('fields', {})
                        episode_title = episode_fields.get('Episode Title', '')
                        calculation = episode_fields.get('Calculation', '')
                        summary = episode_fields.get('Summary', '')
                        ai_summary = episode_fields.get('AI Summary', '')

                        episode_content = (
                            f"Episode Title: {episode_title}\n"
                            f"Episode ID: {calculation}\n"
                            f"Summary:\n{summary}\n{ai_summary}\n"
                            "End of Episode\n\n")
                        google_docs_client.append_to_document(
                            google_doc_id, episode_content)
                        logger.info(
                            f"Appended episode info for Episode ID {calculation} to doc '{google_doc_name}'."
                        )
                        episode_summaries += episode_content

                else:
                    # Document already exists
                    logger.info(
                        f"'{google_doc_name}' already exists. Retrieving existing doc content."
                    )
                    google_doc_id = doc_search_result[1]
                    episode_summaries = google_docs_client.get_document_content(
                        google_doc_id)

                    # Update the podcast record to store doc ID if needed
                    update_fields = {'PodcastEpisodeInfo': google_doc_id}
                    airtable_client.update_record(PODCASTS_TABLE_NAME,
                                                  podcast_id, update_fields)
                    logger.info(
                        f"Existing doc ID {google_doc_id} has been updated in Airtable."
                    )

                    # --------------------------------------------------------------------
                    # NEW CHECK: If the doc is empty, append episodes just as if it were new
                    # --------------------------------------------------------------------
                    if not episode_summaries.strip():
                        logger.info(
                            f"Document '{google_doc_name}' is empty. Appending episode data."
                        )

                        episode_ids = podcast_fields.get(
                            'Podcast Episodes', [])
                        if not episode_ids:
                            logger.warning(
                                f"No episodes linked to Podcast record {podcast_id}"
                            )
                            continue

                        episode_summaries = ''
                        logger.info(
                            f"Found {len(episode_ids)} episode(s) for Podcast record {podcast_id}. Appending them to the existing doc."
                        )
                        for episode_id in episode_ids:
                            # Check stop flag before each episode
                            if stop_flag and stop_flag.is_set():
                                logger.info("Stop flag set - stopping episode processing")
                                return

                            episode_record = airtable_client.get_record(
                                PODCAST_EPISODES_TABLE_NAME, episode_id)
                            episode_fields = episode_record.get('fields', {})
                            episode_title = episode_fields.get(
                                'Episode Title', '')
                            calculation = episode_fields.get('Calculation', '')
                            summary = episode_fields.get('Summary', '')
                            ai_summary = episode_fields.get('AI Summary', '')

                            episode_content = (
                                f"Episode Title: {episode_title}\n"
                                f"Episode ID: {calculation}\n"
                                f"Summary:\n{summary}\n{ai_summary}\n"
                                "End of Episode\n\n")
                            google_docs_client.append_to_document(
                                google_doc_id, episode_content)
                            logger.info(
                                f"Appended episode info for Episode ID {calculation} to doc '{google_doc_name}'."
                            )
                            episode_summaries += episode_content
                    else:
                        # Document has content, so do nothing (or log an info)
                        logger.info(
                            f"Document '{google_doc_name}' already contains content. No new episodes appended."
                        )
                    # --------------------------------------------------------------------

                # Check stop flag before Claude analysis
                if stop_flag and stop_flag.is_set():
                    logger.info("Stop flag set - stopping before Claude analysis")
                    return

                # STEP 5: Prepare prompt content and send to Claude
                logger.info(
                    "Generating prompt for Claude to evaluate podcast fit.")
                prompt_path = "prompts/prompt_determine_good_fit.txt"
                prompt = generate_prompt(
                    {
                        'podcast_name': podcast_name,
                        'episode_summaries': episode_summaries,
                        'client_bio': bio,
                        'client_angles': angles
                    }, prompt_path)
                logger.info(
                    f"Prompt generated. Sending to Claude for 'determine fit' analysis."
                )

                # Claude evaluates fit
                fit_response = claude_client.create_message(prompt,workflow='determine_fit', podcast_id=podcast_id)
                logger.info(
                    f"Claude response received. Length: {len(fit_response)} characters."
                )

                # Parse Claude's JSON response
                logger.info("Parsing Claude's response to extract fit status.")
                extract_path = "prompts/prompt_extract_json.txt"
                json_prompt = generate_prompt({'textResponse': fit_response},
                                              extract_path)
                fit_status_raw = openai_service.transform_text_to_structured_data(
                    json_prompt, fit_response, data_type='fit', workflow='determine_fit_tts', podcast_id=podcast_id)  
                logger.info(f"Raw fit status: {fit_status_raw}")

                fit_status_parsed = fit_status_raw.get("Answer")
                logger.info(f"Parsed fit status: '{fit_status_parsed}'")

                # STEP 6: Update the status back in Airtable's Campaign Manager
                logger.info("Updating Airtable with the new fit status.")
                update_fields = {'Status': fit_status_parsed}
                airtable_client.update_record(CAMPAIGN_MANAGER_TABLE_NAME,
                                              cm_record_id, update_fields)
                logger.info(
                    f"Campaign Manager record {cm_record_id} updated with status '{fit_status_parsed}'."
                )

                logger.info(
                    f"Record {cm_record_id} processing complete. Waiting 30s before next record..."
                )
                
                # Check stop flag before sleep
                if stop_flag and stop_flag.is_set():
                    logger.info("Stop flag set - stopping before delay")
                    return
                    
                time.sleep(30)  # Optional delay to avoid rate limits

            except Exception as e:
                # Log any error with this specific record, then move on to next record
                logger.error(
                    f"Error processing Campaign Manager record {cm_record_id}: {str(e)}"
                )
                
                # Check stop flag before continuing to next record
                if stop_flag and stop_flag.is_set():
                    logger.info("Stop flag set - stopping after error")
                    return
                    
                logger.info("Waiting 30s before moving to next record...")
                time.sleep(30)  # Optional delay
                continue

    except Exception as e:
        # Log a high-level error if initialization or main steps fail
        logger.error(f"Error in determine_fit function: {str(e)}")
        raise


if __name__ == "__main__":
    determine_fit()
