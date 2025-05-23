import logging
from airtable_service import PodcastService
from openai_service import OpenAIService
from external_api_service import ListenNote, PodscanFM
from data_processor import DataProcessor, parse_date
from datetime import datetime, timedelta
from file_manipulation import read_txt_file
import threading
from typing import Optional
import time

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

genre_id_prompt = read_txt_file(
    r"prompts/podcast_search/listennotes_genre_id_prompt.txt")

def generate_genre_ids(openai_service, run_keyword, record_id=None):
    """
    Helper function to generate genre IDs using OpenAI.
    
    Args:
        openai_service: Instance of OpenAIService
        run_keyword: The keyword to search for
        record_id: Optional record ID for tracking
        
    Returns:
        str: Comma-separated genre IDs
    """
    logger.info("Calling OpenAI to generate genre IDs...")
    
    prompt = f"""
    User Search Query:
    "{run_keyword}"

    Provide the list of genre IDs as per the example above. 
    Return the response in JSON format with an 'ids' key containing an array of integers.
    Do not include backticks i.e ```json
    Example JSON Output Format: {{"ids": "139,144,157,99,90,77,253,69,104,84"}}
    """
    
    genre_ids = openai_service.create_chat_completion(
        system_prompt=genre_id_prompt,
        prompt=prompt,
        workflow="generate_genre_ids",
        parse_json=True,
        json_key="ids",
        podcast_id=record_id  # Pass the record_id as podcast_id for tracking
    )
    logger.info(f"Genre IDs generated: {genre_ids}")
    return genre_ids

def process_mipr_podcast_search_listennotes(record_id, stop_flag: Optional[threading.Event] = None):
    # Initialize services
    airtable_service = PodcastService()
    openai_service = OpenAIService()
    external_api_service = ListenNote()
    data_processor = DataProcessor()
    podscan_service = PodscanFM()

    try:
        # Check if we should stop
        if stop_flag and stop_flag.is_set():
            logger.info(f"Stopping ListenNotes podcast search for record {record_id} due to stop flag")
            return
            
        # Fetch the campaign record from Airtable
        logger.info(f"Fetching record {record_id} from Airtable...")
        campaign_record = airtable_service.get_record('Campaigns',
                                                      record_id)['fields']
        logger.info("Record successfully retrieved.")
    except Exception as e:
        logger.error(f"Error retrieving record from Airtable: {e}")
        raise

    try:
        # Extract necessary fields
        run_keyword = campaign_record.get('Run Keyword')
        repeat_number = campaign_record.get('Repeat Number', 1)
        page = campaign_record.get('Page', 1)
        campaign_name = campaign_record.get('CampaignName')

        logger.info("Extracted the main fields from the record.")

        # Ensure necessary fields are available
        if not run_keyword:
            error_msg = f"Campaign {record_id} is missing 'Run Keyword'."
            logger.error(error_msg)
            raise ValueError(error_msg)
    except ValueError as ve:
        logger.error(f"ValueError encountered: {ve}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during field extraction: {e}")
        raise

    # Check stop flag before potentially expensive API call
    if stop_flag and stop_flag.is_set():
        logger.info(f"Stopping ListenNotes podcast search before generating genre IDs for record {record_id} due to stop flag")
        return

    try:
        # Use OpenAI to generate genre IDs
        genre_ids = generate_genre_ids(openai_service, run_keyword, record_id)
    except Exception as e:
        logger.error(f"Error generating genre IDs: {e}")
        raise

    # Loop over the number of repeats
    for i in range(int(repeat_number)):
        # Check stop flag at the start of each repeat loop iteration
        if stop_flag and stop_flag.is_set():
            logger.info(f"Stopping ListenNotes podcast search loop for record {record_id} at repeat {i} due to stop flag")
            return

        offset = (i * 10) + ((int(page) - 1) * 10)
        search_results = None
        max_retries = 3
        base_sleep_time = 1 # seconds

        for attempt in range(max_retries):
             # Check stop flag before each attempt
             if stop_flag and stop_flag.is_set():
                  logger.info(f"Stopping ListenNotes podcast search retry loop for record {record_id} due to stop flag")
                  return
             try:
                # Call the external API
                logger.info(
                    f"Searching ListenNotes podcasts (attempt {attempt + 1}/{max_retries}) with keyword '{run_keyword}' for offset {offset}..."
                )
                search_results = external_api_service.search_podcasts(
                    query=run_keyword, genre_ids=genre_ids, offset=offset)
                logger.info("Podcast search results retrieved.")
                #print(search_results)
                break # Success, exit retry loop
             except Exception as e:
                # Check if it's a rate limit error (429)
                if "Listen Notes API Error: 429" in str(e):
                     if attempt < max_retries - 1:
                          sleep_time = base_sleep_time * (2 ** attempt)
                          logger.warning(f"Rate limit hit (429) on attempt {attempt + 1}. Retrying in {sleep_time} seconds...")
                          # Check stop flag before sleeping
                          if stop_flag and stop_flag.is_set():
                               logger.info(f"Stopping ListenNotes podcast search during rate limit backoff for record {record_id} due to stop flag")
                               return
                          time.sleep(sleep_time)
                          # Check stop flag after sleeping
                          if stop_flag and stop_flag.is_set():
                               logger.info(f"Stopping ListenNotes podcast search after rate limit backoff for record {record_id} due to stop flag")
                               return
                     else:
                          logger.error(f"Rate limit error (429) persisted after {max_retries} attempts. Aborting search for this offset.")
                          # Raise the last exception or handle as needed (e.g., break outer loop)
                          raise e # Re-raise the final exception
                else:
                     # Not a rate limit error, re-raise immediately
                     logger.error(f"Non-rate-limit error searching podcasts: {e}")
                     raise e # Re-raise other exceptions

        # If search_results is still None after retries (shouldn't happen if exception raised, but safer)
        if search_results is None:
             logger.error(f"Failed to retrieve search results for offset {offset} after retries, skipping processing for this offset.")
             continue # Skip to the next offset

        # Process and update Airtable records
        results_list = search_results.get('results', [])
        logger.info(f"Found {len(results_list)} results from ListenNotes for offset {offset}. Processing...") # Log count

        for result in results_list:
            # Check stop flag inside the results loop
            if stop_flag and stop_flag.is_set():
                logger.info(f"Stopping ListenNotes podcast result processing for record {record_id} due to stop flag")
                return

            # Log basic info for each result before email check
            podcast_title_ln = result.get('title_original', '[No Title]')
            has_email = bool(result.get('email'))
            logger.info(f"Processing result: '{podcast_title_ln}' | Has Email: {has_email}")

            if has_email:
                podscan_podcast_id = None
                rss_url = result.get('rss')

                # Attempt to find matching podcast on Podscan via RSS
                if rss_url:
                    # Check stop flag before Podscan RSS search
                    if stop_flag and stop_flag.is_set():
                        logger.info(f"Stopping ListenNotes podcast processing before Podscan RSS check for record {record_id} due to stop flag")
                        return
                    try:
                        logger.info(f"Checking Podscan for RSS: {rss_url}")
                        podscan_results = podscan_service.search_podcast_by_rss(rss_url)
                        # Check if results is a list and not empty
                        if isinstance(podscan_results, list) and podscan_results:
                            podscan_podcast_id = podscan_results[0].get('podcast_id')
                            logger.info(f"Found matching Podscan ID: {podscan_podcast_id}")
                        else:
                            # Log if empty list or unexpected type
                            logger.info(f"No matching podcast found on Podscan for RSS: {rss_url} (Result: {podscan_results})")
                            podscan_podcast_id = None # Ensure it's None if not found or error
                    except Exception as rss_err:
                        logger.warning(f"Error searching Podscan by RSS ({rss_url}): {rss_err}")
                        podscan_podcast_id = None # Ensure it's None on exception

                # Check stop flag before Airtable update/create
                if stop_flag and stop_flag.is_set():
                    logger.info(f"Stopping ListenNotes podcast processing before Airtable update for record {record_id} due to stop flag")
                    return
                try:
                    # Pass the potential Podscan ID to the processor
                    data_processor.process_podcast_result_with_listennotes(
                        result, record_id, campaign_name, airtable_service, podscan_podcast_id=podscan_podcast_id
                    )
                    logger.info(
                        "Processed podcast result and updated Airtable.")
                except Exception as e:
                    logger.error(f"Error processing podcast result: {e}")
                    # Decide if you want to continue or re-raise here.
                    # raise

    # Update the campaign record in Airtable
    try:
        # Check stop flag before final campaign update
        if stop_flag and stop_flag.is_set():
            logger.info(f"Stopping ListenNotes campaign update for record {record_id} due to stop flag")
            return
            
        new_page = int(page) + int(repeat_number)
        airtable_service.update_record('Campaigns', record_id,
                                       {'Page': new_page})
        logger.info(f"Updated campaign record with new page value: {new_page}")
    except Exception as e:
        logger.error(f"Error updating campaign record in Airtable: {e}")
        raise


def process_mipr_podcast_search_with_podscan(record_id, stop_flag: Optional[threading.Event] = None):
    # Initialize services
    airtable_service = PodcastService()
    openai_service = OpenAIService()
    podscan_service = PodscanFM()
    data_processor = DataProcessor()

    try:
        # Check if we should stop
        if stop_flag and stop_flag.is_set():
            logger.info(f"Stopping PodScan podcast search for record {record_id} due to stop flag")
            return
            
        # Fetch the campaign record from Airtable
        logger.info(f"Fetching record {record_id} from Airtable...")
        campaign_record = airtable_service.get_record('Campaigns',
                                                      record_id)['fields']
        logger.info("Record successfully retrieved.")
    except Exception as e:
        logger.error(f"Error retrieving record from Airtable: {e}")
        raise
    try:
        # Extract necessary fields
        run_keyword = campaign_record.get('Run Keyword')
        repeat_number = campaign_record.get('Repeat Number', 1)
        page = campaign_record.get('Page', 1)
        campaign_name = campaign_record.get('CampaignName')

        logger.info("Extracted the main fields from the record.")

        # Ensure necessary fields are available
        if not run_keyword:
            error_msg = f"Campaign {record_id} is missing 'Run Keyword'."
            logger.error(error_msg)
            raise ValueError(error_msg)
    except ValueError as ve:
        logger.error(f"ValueError encountered: {ve}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during field extraction: {e}")
        raise

    try:
        # Check if we should stop
        if stop_flag and stop_flag.is_set():
            logger.info(f"Stopping PodScan podcast search for record {record_id} due to stop flag")
            return
            
        # Use OpenAI to generate genre IDs
        category_ids = generate_genre_ids(openai_service, run_keyword, record_id)
    except Exception as e:
        logger.error(f"Error generating genre IDs: {e}")
        raise

    # Loop over the number of repeats
    for i in range(int(repeat_number)):
        # Check if we should stop
        if stop_flag and stop_flag.is_set():
            logger.info(f"Stopping PodScan podcast search for record {record_id} due to stop flag")
            return
            
        try:
            podcasts = podscan_service.search_podcasts(
                run_keyword, category_id=category_ids, page=page)
            if not podcasts:
                print(f"No podcasts found on page {page}.")
                break
            logger.info("Podcast search results retrieved.")
        except Exception as e:
            logger.error(f"Error searching podcasts: {e}")
            raise

        for podcast in podcasts:
            # Check if we should stop
            if stop_flag and stop_flag.is_set():
                logger.info(f"Stopping PodScan podcast processing for record {record_id} due to stop flag")
                return
                
            if podcast.get("email"):
                try:
                    dt_published = parse_date(podcast.get("last_posted_at"))

                    if dt_published is None:
                        logger.warning(
                            f"Could not parse published date for podcast '{podcast.get('podcast_name')}'."
                        )
                        continue

                    # Check if the episode was released within the past year
                    time_frame = datetime.now(
                        dt_published.tzinfo) - timedelta(days=90)
                    if dt_published < time_frame:
                        logger.info(
                            f"Episode '{podcast.get('podcast_name')}' is older than 3 months. Skipping update."
                        )
                        continue

                    
                    podcast["last_posted_at"] = dt_published.strftime('%Y-%m-%d')
                    
                    data_processor.process_podcast_result_with_podscan(
                        podcast, record_id, campaign_name, airtable_service)
                    logger.info(
                        "Processed podcast result and updated Airtable.")
                except Exception as e:
                    logger.error(f"Error processing podcast result: {e}")

        page += 1

    # Update the campaign record in Airtable
    try:
        # Check if we should stop
        if stop_flag and stop_flag.is_set():
            logger.info(f"Stopping PodScan campaign update for record {record_id} due to stop flag")
            return
            
        new_page = int(page)
        airtable_service.update_record('Campaigns', record_id,
                                       {'Page': new_page})
        logger.info(f"Updated campaign record with new page value: {new_page}")
    except Exception as e:
        logger.error(f"Error updating campaign record in Airtable: {e}")
        raise
