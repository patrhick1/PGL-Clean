import logging
from airtable_service import PodcastService
from openai_service import OpenAIService
from external_api_service import ListenNote
from data_processor import DataProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def process_mipr_podcast_search(record_id):
    # Initialize services
    airtable_service = PodcastService()
    openai_service = OpenAIService()
    external_api_service = ListenNote()
    data_processor = DataProcessor()

    try:
        # Fetch the campaign record from Airtable
        logger.info(f"Fetching record {record_id} from Airtable...")
        campaign_record = airtable_service.get_record('Campaigns', record_id)['fields']
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
        # Use OpenAI to generate genre IDs
        logger.info("Calling OpenAI to generate genre IDs...")
        genre_ids = openai_service.generate_genre_ids(run_keyword)
        logger.info(f"Genre IDs generated: {genre_ids}")
    except Exception as e:
        logger.error(f"Error generating genre IDs: {e}")
        raise

    # Loop over the number of repeats
    for i in range(int(repeat_number)):
        offset = (i * 10) + ((int(page) - 1) * 10)
        try:
            # Call the external API
            logger.info(
                f"Searching podcasts with keyword '{run_keyword}' for offset {offset}..."
            )
            search_results = external_api_service.search_podcasts(
                query=run_keyword,
                genre_ids=genre_ids,
                offset=offset
            )
            logger.info("Podcast search results retrieved.")
            #print(search_results)
        except Exception as e:
            logger.error(f"Error searching podcasts: {e}")
            # Depending on your needs, you might skip the rest or re-raise.
            raise

        # Process and update Airtable records
        for result in search_results.get('results', []):
            try:
                data_processor.process_podcast_result(result, record_id, campaign_name, airtable_service)
                logger.info("Processed podcast result and updated Airtable.")
            except Exception as e:
                logger.error(f"Error processing podcast result: {e}")
                # Decide if you want to continue or re-raise here.
                # raise

       # Update the campaign record in Airtable
    try:
        new_page = int(page) + int(repeat_number)
        airtable_service.update_record('Campaigns', record_id, {'Page': new_page})
        logger.info(f"Updated campaign record with new page value: {new_page}")
    except Exception as e:
        logger.error(f"Error updating campaign record in Airtable: {e}")
        raise


if __name__ == "__main__":
    logger.info("Process started")
    test_record_id = "recz7C3FJe915NMfi"
    try:
        process_mipr_podcast_search(test_record_id)
        logger.info("Process ended successfully.")
    except Exception as e:
        logger.error(f"Process ended with an error: {e}")
