import logging
from airtable_service import PodcastService
from openai_service import OpenAIService
from external_api_service import ListenNote, PodscanFM
from data_processor import DataProcessor, parse_date
from datetime import datetime, timedelta
from file_manipulation import read_txt_file
import threading
from typing import Optional

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

    try:
        # Check if we should stop
        if stop_flag and stop_flag.is_set():
            logger.info(f"Stopping ListenNotes podcast search for record {record_id} due to stop flag")
            return
            
        # Use OpenAI to generate genre IDs
        genre_ids = generate_genre_ids(openai_service, run_keyword, record_id)
    except Exception as e:
        logger.error(f"Error generating genre IDs: {e}")
        raise

    # Loop over the number of repeats
    for i in range(int(repeat_number)):
        # Check if we should stop
        if stop_flag and stop_flag.is_set():
            logger.info(f"Stopping ListenNotes podcast search for record {record_id} due to stop flag")
            return
            
        offset = (i * 10) + ((int(page) - 1) * 10)
        try:
            # Call the external API
            logger.info(
                f"Searching podcasts with keyword '{run_keyword}' for offset {offset}..."
            )
            search_results = external_api_service.search_podcasts(
                query=run_keyword, genre_ids=genre_ids, offset=offset)
            logger.info("Podcast search results retrieved.")
            #print(search_results)
        except Exception as e:
            logger.error(f"Error searching podcasts: {e}")
            # Depending on your needs, you might skip the rest or re-raise.
            raise

        # Process and update Airtable records
        for result in search_results.get('results', []):
            # Check if we should stop
            if stop_flag and stop_flag.is_set():
                logger.info(f"Stopping ListenNotes podcast search processing for record {record_id} due to stop flag")
                return
                
            if result.get('email'):
                try:
                    data_processor.process_podcast_result_with_listennotes(
                        result, record_id, campaign_name, airtable_service)
                    logger.info(
                        "Processed podcast result and updated Airtable.")
                except Exception as e:
                    logger.error(f"Error processing podcast result: {e}")
                    # Decide if you want to continue or re-raise here.
                    # raise

    # Update the campaign record in Airtable
    try:
        # Check if we should stop
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
