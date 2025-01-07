import logging
from airtable_service import PodcastService
from anthropic_service import AnthropicService
from dotenv import load_dotenv
from google_docs_service import GoogleDocsService
import os
import time
from data_processor import generate_prompt, convert_into_json_format

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def determine_fit():
    logging.info('Starting Determine Fit Automation')

    # Initialize the services
    airtable_client = PodcastService()
    claude_client = AnthropicService()
    google_docs_client = GoogleDocsService()

    PODCAST_INFO_FOLDER_ID = os.getenv('GOOGLE_PODCAST_INFO_FOLDER_ID')

    # Define the Airtable table names
    CAMPAIGN_MANAGER_TABLE_NAME = 'Campaign Manager'  # Adjust as needed
    CAMPAIGNS_TABLE_NAME = 'Campaigns'
    PODCASTS_TABLE_NAME = 'Podcasts'
    PODCAST_EPISODES_TABLE_NAME = 'Podcast_Episodes'

    #Define views in podcast table
    OUTREACH_READY_VIEW = 'OR'

    # Step 1: Fetch records from Campaign Manager from the OR View"

    campaign_manager_records = airtable_client.get_records_from_view(CAMPAIGN_MANAGER_TABLE_NAME, OUTREACH_READY_VIEW)

    # Process each record
    for cm_record in campaign_manager_records:
        try:
            cm_record_id = cm_record['id']
            cm_fields = cm_record.get('fields', {})

            # Get Campaign record linked in 'Campaigns' field
            campaign_ids = cm_fields.get('Campaigns', [])
            if not campaign_ids:
                logging.warning(f"No campaign linked to Campaign Manager record {cm_record_id}")
                continue
            campaign_id = campaign_ids[0]  # Assuming single campaign

            campaign_record = airtable_client.get_record(CAMPAIGNS_TABLE_NAME, campaign_id)
            campaign_fields = campaign_record.get('fields', {})
            bio = campaign_fields.get('TextBio', '')
            angles = campaign_fields.get('TextAngles', '')

            # Get Podcast record linked in 'Podcast Name' field
            podcast_ids = cm_fields.get('Podcast Name', [])
            if not podcast_ids:
                logging.warning(f"No podcast linked to Campaign Manager record {cm_record_id}")
                continue
            podcast_id = podcast_ids[0]  # Assuming single podcast

            podcast_record = airtable_client.get_record(PODCASTS_TABLE_NAME, podcast_id)
            podcast_fields = podcast_record.get('fields', {})
            podcast_name = podcast_fields.get('Podcast Name', '')



            # Create Google Doc named '{{Podcast Name}} - Info' in specified folder
            google_doc_name = f"{podcast_name} - Info"

            search_file = google_docs_client.check_file_exists_in_folder(google_doc_name)

            if search_file[0] == False:

                google_doc_id = google_docs_client.create_document_without_content(google_doc_name, PODCAST_INFO_FOLDER_ID)

                # Update Podcast record with 'PodcastEpisodeInfo' field set to Google Doc ID
                update_fields = {
                    'PodcastEpisodeInfo': google_doc_id
                }
                airtable_client.update_record(PODCASTS_TABLE_NAME, podcast_id, update_fields)


                # Get 'Podcast Episodes' from Podcast record
                episode_ids = podcast_fields.get('Podcast Episodes', [])
                if not episode_ids:
                    logging.warning(f"No episodes linked to Podcast record {podcast_id}")
                    continue

                # For each 'Podcast Episode', get the Episode record and append details to the Google Doc
                episode_summaries = ''
                for episode_id in episode_ids:
                    episode_record = airtable_client.get_record(PODCAST_EPISODES_TABLE_NAME, episode_id)
                    episode_fields = episode_record.get('fields', {})

                    episode_title = episode_fields.get('Episode Title', '')
                    calculation = episode_fields.get('Calculation', '')
                    summary = episode_fields.get('Summary', '')
                    ai_summary = episode_fields.get('AI Summary', '')

                    episode_content = f"""Episode Title: {episode_title}
    Episode ID: {calculation}
    Summary:
    {summary}
    {ai_summary}
    End of Episode

    """
                    # Append to Google Doc
                    google_docs_client.append_to_document(google_doc_id, episode_content)

                    # Collect summaries
                    episode_summaries += episode_content

            elif search_file[0] == True:
                print(f"{google_doc_name} already exist in the folder")
                google_doc_id = search_file[1]
                episode_summaries = google_docs_client.get_document_content(google_doc_id)
                # Update Podcast record with 'PodcastEpisodeInfo' field set to Google Doc ID
                update_fields = {
                    'PodcastEpisodeInfo': google_doc_id
                }
                airtable_client.update_record(PODCASTS_TABLE_NAME, podcast_id, update_fields)

            # Now, prepare prompt to send to Claude
            prompt_path = r"app\prompts\prompt_determine_good_fit.txt"
            prompt = generate_prompt({'podcast_name': podcast_name,'episode_summaries':episode_summaries,
                                             'client_bio': bio,'client_angles': angles}, prompt_path)

            # Send prompt to Claude
            podcast_fit_claude_response = claude_client.create_message(prompt)

            # Parse response from Claude
            prompt_path = r"app\prompts\prompt_extract_json.txt"
            prompt = generate_prompt({'textResponse':podcast_fit_claude_response}, prompt_path)
            fit_status = claude_client.create_message(prompt)
            fit_status = convert_into_json_format(fit_status).get("Answer")

            # Update 'Status' field in Campaign Manager record
            update_fields = {
                'Status': fit_status
            }
            

            airtable_client.update_record(CAMPAIGN_MANAGER_TABLE_NAME, cm_record_id, update_fields)
            print(f"The Status of Record ID {cm_record_id} has been updated in {CAMPAIGN_MANAGER_TABLE_NAME} table to {fit_status}")

            logging.info(f"Processed Campaign Manager record {cm_record_id}, status updated to '{fit_status}'")  
            time.sleep(30)
        except Exception as e:
            logging.error(f"Error processing Campaign Manager record {cm_record_id}: {str(e)}")
            time.sleep(30)
            continue



if __name__ == "__main__":
    determine_fit()
    logger.info("Process finished.")