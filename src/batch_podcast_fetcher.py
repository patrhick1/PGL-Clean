import logging
import os
import time
import threading
import argparse
from typing import Optional

# Assuming these modules are in the same directory or accessible via PYTHONPATH
from airtable_service import PodcastService
from openai_service import OpenAIService
from google_docs_service import GoogleDocsService
from external_api_service import ListenNote, PodscanFM
from data_processor import DataProcessor
from mipr_podcast import generate_genre_ids # Re-use the genre ID generator

# --- Configuration ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
# Field name in Airtable 'Campaigns' table containing comma-separated keywords
KEYWORDS_FIELD_NAME = "Search Keywords"
keyword_prompt_doc_id = '18r8jTqj5cCzhnlajjKTPCJ4roG7kxTTNgCGAlW5WxoM'

def generate_keywords(campaign_fields, keyword_prompt_doc_id, google_docs_service, openai_service):

    keyword_prompt_content = google_docs_service.get_document_content(keyword_prompt_doc_id)
    
    # Generate keywords using OpenAI
    prompt_for_keywords = f"""
        Bio: 
        {campaign_fields.get("TextBio")}

        Angles:
        {campaign_fields.get("TextAngles")}
    """
    
    logger.info("Generating keywords using OpenAI")

    keywords = openai_service.create_chat_completion(
            keyword_prompt_content, 
            prompt_for_keywords,
            workflow="bio_and_angles"
        )
    
    logger.info(f"Generated {len(keywords)} characters of keywords")
    return keywords
    

def process_campaign_keywords(campaign_record_id: str, stop_flag: Optional[threading.Event] = None):
    """
    Fetches a campaign record, extracts keywords, and searches ListenNotes exhaustively
    (handling pagination) for each keyword.

    Args:
        campaign_record_id: The Airtable record ID of the campaign to process.
        stop_flag: Optional threading.Event to signal graceful shutdown.
    """
    logger.info(f"Starting batch podcast fetching process for Campaign ID: {campaign_record_id}")

    # --- Initialize Services ---
    try:
        airtable_service = PodcastService()
        openai_service = OpenAIService()
        external_api_service = ListenNote()
        data_processor = DataProcessor()
        podscan_service = PodscanFM()
        google_docs_service = GoogleDocsService()
        logger.info("Services initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        return # Cannot proceed without services

    # --- Fetch Campaign Record ---
    try:
        # Check stop flag before fetching
        if stop_flag and stop_flag.is_set():
            logger.info("Stopping process before fetching campaign record.")
            return

        logger.info(f"Fetching campaign record {campaign_record_id} from Airtable...")
        campaign_record_data = airtable_service.get_record('Campaigns', campaign_record_id)
        if not campaign_record_data or 'fields' not in campaign_record_data:
            logger.error(f"Could not retrieve valid record data for Campaign ID: {campaign_record_id}")
            return
        campaign_fields = campaign_record_data['fields']
        logger.info("Campaign record retrieved successfully.")

    except Exception as e:
        logger.error(f"Error retrieving campaign record {campaign_record_id} from Airtable: {e}")
        return # Cannot proceed without the record

    # --- Extract Keywords and Campaign Info ---
    try:
        campaign_name = campaign_fields.get('CampaignName', f"Campaign_{campaign_record_id}") # Use ID as fallback name
        keywords_str = generate_keywords(campaign_fields, keyword_prompt_doc_id, google_docs_service, openai_service)

        if not keywords_str:
            logger.warning(f"Campaign '{campaign_name}' (ID: {campaign_record_id}) has no keywords in the '{KEYWORDS_FIELD_NAME}' field. Skipping.")
            return

        # Split keywords string into a list, stripping whitespace
        keywords = [keyword.strip() for keyword in keywords_str.split(',') if keyword.strip()]

        if not keywords:
            logger.warning(f"Keywords field for Campaign '{campaign_name}' was present but contained no valid keywords after splitting. Skipping.")
            return

        logger.info(f"Found {len(keywords)} keywords for Campaign '{campaign_name}': {keywords}")

    except Exception as e:
        logger.error(f"Error extracting keywords or campaign name for {campaign_record_id}: {e}")
        return

    # --- Process Each Keyword ---
    total_processed_results_overall = 0
    for keyword in keywords:
        # Check stop flag at the start of each keyword loop
        if stop_flag and stop_flag.is_set():
            logger.info(f"Stopping processing before keyword '{keyword}' for Campaign '{campaign_name}'.")
            break # Exit the keyword loop

        logger.info(f"--- Processing keyword: '{keyword}' for Campaign '{campaign_name}' ---")
        keyword_processed_count = 0
        current_offset = 0
        has_more_results = True

        # 1. Generate Genre IDs for the current keyword (done once per keyword)
        try:
             logger.info(f"Generating genre IDs for keyword: '{keyword}'...")
             genre_ids = generate_genre_ids(openai_service, keyword, campaign_record_id)
             if not genre_ids:
                 logger.warning(f"Could not generate genre IDs for keyword '{keyword}'. Skipping keyword.")
                 continue # Move to the next keyword
        except Exception as e:
             logger.error(f"Error generating genre IDs for keyword '{keyword}': {e}")
             continue # Move to the next keyword

        # 2. Inner Loop: Fetch and process paginated results for the CURRENT keyword
        while has_more_results:
            # Check stop flag at the start of each page fetch
            if stop_flag and stop_flag.is_set():
                logger.info(f"Stopping pagination loop for keyword '{keyword}'.")
                has_more_results = False # Ensure loop termination
                break # Exit pagination loop

            search_results_data = None
            max_retries = 3
            base_sleep_time = 1

            # Retry logic for fetching a specific page/offset
            for attempt in range(max_retries):
                 if stop_flag and stop_flag.is_set(): break # Check before attempt
                 try:
                    logger.info(f"Searching ListenNotes (attempt {attempt + 1}/{max_retries}) for keyword '{keyword}' (offset {current_offset})...")
                    # *** Make the API call ***
                    search_results_data = external_api_service.search_podcasts(
                        query=keyword,
                        genre_ids=genre_ids,
                        offset=current_offset
                    )
                    logger.info(f"ListenNotes search successful for keyword '{keyword}', offset {current_offset}.")
                    break # Success
                 except Exception as e:
                    if "Listen Notes API Error: 429" in str(e):
                         if attempt < max_retries - 1:
                              sleep_time = base_sleep_time * (2 ** attempt)
                              logger.warning(f"Rate limit hit (429) for keyword '{keyword}', offset {current_offset}. Retrying in {sleep_time} seconds...")
                              if stop_flag and stop_flag.is_set(): break
                              time.sleep(sleep_time)
                              if stop_flag and stop_flag.is_set(): break
                         else:
                              logger.error(f"Rate limit error (429) persisted for keyword '{keyword}', offset {current_offset}. Stopping pagination for this keyword.")
                              search_results_data = None
                              has_more_results = False # Stop pagination for this keyword on persistent failure
                              break
                    else:
                         logger.error(f"Error searching ListenNotes for keyword '{keyword}', offset {current_offset}: {e}")
                         search_results_data = None
                         has_more_results = False # Stop pagination on other errors too
                         break

            # Check if retry loop was stopped or failed
            if (stop_flag and stop_flag.is_set()) or not has_more_results:
                break # Exit pagination loop

            # If search failed for this offset, stop pagination for this keyword
            if search_results_data is None or 'results' not in search_results_data:
                 logger.error(f"Failed to retrieve valid search results from ListenNotes for keyword '{keyword}', offset {current_offset}. Stopping pagination for this keyword.")
                 has_more_results = False
                 break # Exit pagination loop

            # 3. Process the current page of results
            results_list = search_results_data.get('results', [])
            if not results_list:
                 logger.info(f"No more results found for keyword '{keyword}' at offset {current_offset}.")
                 has_more_results = False # No results means we're done
                 break

            logger.info(f"Processing {len(results_list)} results for keyword '{keyword}', offset {current_offset}...")
            page_processed_count = 0
            for result in results_list:
                if stop_flag and stop_flag.is_set():
                    logger.info("Stopping processing of search results.")
                    has_more_results = False # Ensure loop terminates
                    break # Exit results processing loop

                # --- Start: Added Email Filter and Podscan Check ---
                if result.get('email'):
                    podscan_podcast_id = None
                    rss_url = result.get('rss')
                    podcast_title_ln = result.get('title_original', '[No Title]')

                    # Attempt to find matching podcast on Podscan via RSS
                    if rss_url:
                        # Check stop flag before Podscan RSS search
                        if stop_flag and stop_flag.is_set():
                            logger.info(f"Stopping batch podcast processing before Podscan RSS check for campaign {campaign_record_id} due to stop flag")
                            has_more_results = False # Ensure outer loops terminate
                            break # Exit results processing loop
                        try:
                            logger.info(f"Checking Podscan for '{podcast_title_ln}' via RSS: {rss_url}")
                            podscan_results = podscan_service.search_podcast_by_rss(rss_url)
                            # Check if results is a list and not empty
                            if isinstance(podscan_results, list) and podscan_results:
                                podscan_podcast_id = podscan_results[0].get('podcast_id')
                                logger.info(f"Found matching Podscan ID for '{podcast_title_ln}': {podscan_podcast_id}")
                            else:
                                logger.info(f"No matching podcast found on Podscan for '{podcast_title_ln}' using RSS: {rss_url}")
                                podscan_podcast_id = None # Ensure it's None if not found
                        except Exception as rss_err:
                            logger.warning(f"Error searching Podscan by RSS ({rss_url}) for '{podcast_title_ln}': {rss_err}")
                            podscan_podcast_id = None # Ensure it's None on exception

                    # Check stop flag before Airtable update/create
                    if stop_flag and stop_flag.is_set():
                         logger.info(f"Stopping batch podcast processing before Airtable update for campaign {campaign_record_id} due to stop flag")
                         has_more_results = False # Ensure outer loops terminate
                         break # Exit results processing loop

                    try:
                        # Pass the potential Podscan ID to the processor
                        data_processor.process_podcast_result_with_listennotes(
                            result,
                            campaign_record_id,
                            campaign_name,
                            airtable_service,
                            podscan_podcast_id=podscan_podcast_id # Pass Podscan ID
                        )
                        page_processed_count += 1
                    except Exception as e:
                        logger.error(f"Error processing result '{podcast_title_ln}' (with email) for keyword '{keyword}': {e}")
                else:
                    # Log podcasts skipped due to missing email
                    podcast_title_skipped = result.get('title_original', '[No Title]')
                    logger.debug(f"Skipping podcast '{podcast_title_skipped}' for keyword '{keyword}' because it has no email.")
                # --- End: Added Email Filter and Podscan Check ---

            keyword_processed_count += page_processed_count
            logger.info(f"Finished processing page for offset {current_offset} ({page_processed_count} results processed with email).")

            # 4. Check pagination for next iteration
            has_next = search_results_data.get('has_next', False)
            next_offset = search_results_data.get('next_offset', None)

            if has_next and next_offset is not None:
                 logger.info(f"More results available for keyword '{keyword}'. Next offset: {next_offset}")
                 current_offset = next_offset
                 # Optional: Add a small delay between pages
                 if not stop_flag or not stop_flag.is_set():
                     time.sleep(1) # Sleep 1 sec between pages
            else:
                 logger.info(f"No more pages indicated for keyword '{keyword}'.")
                 has_more_results = False # Stop the pagination loop

            # End of inner pagination (while) loop

        logger.info(f"Finished processing keyword '{keyword}'. Total results processed for this keyword: {keyword_processed_count}")
        total_processed_results_overall += keyword_processed_count

        # Check stop flag after finishing a keyword
        if stop_flag and stop_flag.is_set():
            logger.info("Stopping processing after finishing keyword.")
            break # Exit the main keyword loop

        # Optional: Add a longer delay between keywords
        if not stop_flag or not stop_flag.is_set():
             time.sleep(3) # Sleep 3 seconds between keywords
        # End of outer keyword (for) loop

    logger.info(f"--- Batch fetching complete for Campaign '{campaign_name}' (ID: {campaign_record_id}) ---")
    logger.info(f"Total results processed across all keywords: {total_processed_results_overall}")

    # --- Update Campaign Status (Optional) ---
    # Consider updating a status field in the 'Campaigns' table here
    # e.g., airtable_service.update_record('Campaigns', campaign_record_id, {'Batch Fetch Status': 'Completed', 'Last Batch Fetch': datetime.now().isoformat()})


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch podcasts exhaustively from ListenNotes based on keywords in an Airtable Campaign record.") # Updated description
    parser.add_argument("record_id", help="The Airtable record ID of the Campaign to process.")
    args = parser.parse_args()

    logger.info(f"Script started for Campaign Record ID: {args.record_id}")

    # Example of running without threading stop flag
    try:
        process_campaign_keywords(args.record_id)
    except Exception as e:
        logger.error(f"An unexpected error occurred during script execution: {e}", exc_info=True)

    logger.info(f"Script finished for Campaign Record ID: {args.record_id}") 