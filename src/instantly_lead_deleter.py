import os
import logging
import time
from typing import List, Dict, Optional
from dotenv import load_dotenv

# --- Add project root to sys.path ---
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # Goes up one level from 'src' to 'PGL'
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End sys.path modification ---

# Assuming these are in the same src directory or PYTHONPATH is set up
from src.external_api_service import InstantlyAPI
from src.instantly_leads_db import get_instantly_lead_by_id, get_db_connection # For backup check

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Define which campaign IDs to process. Can be moved to env vars or a config file.
# For testing, you might want to use a specific campaign ID.
CAMPAIGN_IDS_TO_PROCESS = [
    "afe3a4d7-5ed7-4fd4-9f8f-cf4e2ddc843d", # Example Campaign ID 1
    "d52f85c0-8341-42d8-9e07-99c6b758fa0b", # Example Campaign ID 2
    # Add more campaign IDs here as needed
    "7b4a5386-8fa1-4059-8ded-398c0f48972b",
    "186fcab7-7c86-4086-9278-99238c453470",
    "ae1c1042-d10e-4cfc-ba4c-743a42550c85",
    "ccbd7662-bbed-46ee-bd8f-1bc374646472",
    "ad2c89bc-686d-401e-9f06-c6ff9d9b7430",
    "3816b624-2a1f-408e-91a9-b9f730d03e2b",
    "60346de6-915c-43fa-9dfa-b77983570359",
    "5b1053b5-8143-4814-a9dc-15408971eac8",
    "02b1d9ff-0afe-4b64-ac15-a886f43bdbce",
    "0725cdd8-b090-4da4-90af-6ca93ac3c267",
    "640a6822-c1a7-48c7-8385-63b0d4c283fc",
    "540b0539-f1c2-4612-94d8-df6fab42c2a7",
    "b55c61b6-262c-4390-b6e0-63dfca1620c2"
]

# Define which statuses are considered safe to delete if the lead is backed up.
# Refer to the Instantly API documentation for exact filter enum values.
# Example: "FILTER_LEAD_LOST", "FILTER_VAL_COMPLETED", "FILTER_VAL_UNSUBSCRIBED", "FILTER_VAL_BOUNCED"
LEAD_STATUSES_TO_DELETE = [
    "FILTER_LEAD_LOST",
    "FILTER_VAL_UNSUBSCRIBED",
    "FILTER_VAL_BOUNCED",
    "FILTER_VAL_INVALID"
]

# Delay between batches of deletions to avoid hitting API rate limits aggressively
DELETION_BATCH_DELAY_SECONDS = 1
# Number of leads to delete in a single batch (Instantly API might have limits on DELETE frequency)
# Keeping it small initially for safety.
DELETION_BATCH_SIZE = 5 


def delete_leads_from_instantly(campaign_ids: List[str], statuses_to_delete: List[str], dry_run: bool = True):
    """
    Fetches leads from specified campaigns with specified statuses from Instantly,
    checks if they are backed up in the local PostgreSQL database, and then deletes them from Instantly.

    Args:
        campaign_ids (List[str]): A list of Instantly campaign IDs to process.
        statuses_to_delete (List[str]): A list of Instantly lead filter statuses to target for deletion.
        dry_run (bool): If True, only logs actions without actually deleting leads. Defaults to True.
    """
    instantly_service = InstantlyAPI()
    total_leads_identified_for_deletion = 0
    total_leads_confirmed_backed_up = 0
    total_leads_successfully_deleted = 0
    total_leads_failed_to_delete = 0
    total_leads_not_backed_up = 0

    logger.info(f"--- Starting Instantly.ai Lead Deletion Process ---")
    logger.info(f"Dry Run: {dry_run}")
    logger.info(f"Campaign IDs to process: {campaign_ids}")
    logger.info(f"Lead statuses to target for deletion: {statuses_to_delete}")

    for campaign_id in campaign_ids:
        logger.info(f"\nProcessing Campaign ID: {campaign_id}")
        for status_filter in statuses_to_delete:
            logger.info(f"  Fetching leads with status '{status_filter}' for campaign '{campaign_id}'...")
            
            leads_to_consider_for_deletion = []
            starting_after = None
            page_num = 1

            while True:
                logger.info(f"    Fetching page {page_num} (starting_after: {starting_after})...")
                try:
                    # Corrected call to list_leads_from_campaign
                    current_page_leads_response = instantly_service.list_leads_from_campaign(
                        campaign_id=campaign_id,
                        filter_str=status_filter, # Pass the status filter
                        limit=100, # Maximize limit to reduce pages, InstantlyAPI method handles pagination internally
                        starting_after=starting_after
                    )
                    # list_leads_from_campaign now returns the list of items directly
                    current_page_leads = current_page_leads_response 

                    if not current_page_leads:
                        logger.info(f"    No more leads found for status '{status_filter}' in campaign '{campaign_id}' on this page or subsequent pages.")
                        break # Break from while True as list_leads_from_campaign handles all pages
                    
                    leads_to_consider_for_deletion.extend(current_page_leads)
                    logger.info(f"    Fetched {len(current_page_leads)} leads for status '{status_filter}'. Total for this status: {len(leads_to_consider_for_deletion)}.")
                    
                    # Since list_leads_from_campaign now fetches all leads for the given filter internally,
                    # we break after the first successful call.
                    break 

                except Exception as e:
                    logger.error(f"    Error fetching leads for campaign {campaign_id} with status {status_filter}: {e}")
                    break # Stop fetching for this status/campaign on error

            if not leads_to_consider_for_deletion:
                logger.info(f"  No leads found with status '{status_filter}' in campaign '{campaign_id}'.")
                continue

            total_leads_identified_for_deletion += len(leads_to_consider_for_deletion)
            logger.info(f"  Identified {len(leads_to_consider_for_deletion)} leads with status '{status_filter}' for potential deletion.")

            deletion_batch_ids = []
            for lead_data in leads_to_consider_for_deletion:
                lead_id = lead_data.get('id')
                lead_email = lead_data.get('email', '[No Email]')

                if not lead_id:
                    logger.warning(f"    Lead data missing ID: {lead_data}")
                    continue

                # Check if lead exists in local backup
                # logger.debug(f"    Checking backup for lead ID: {lead_id} ({lead_email})")
                backed_up_lead = get_instantly_lead_by_id(lead_id)
                
                if backed_up_lead:
                    total_leads_confirmed_backed_up +=1
                    # logger.debug(f"    Lead ID: {lead_id} ({lead_email}) IS backed up.")
                    deletion_batch_ids.append(lead_id)
                else:
                    total_leads_not_backed_up +=1
                    logger.warning(f"    Lead ID: {lead_id} ({lead_email}) is NOT backed up. SKIPPING deletion from Instantly.")

                # Process deletions in batches
                if len(deletion_batch_ids) >= DELETION_BATCH_SIZE:
                    if not dry_run:
                        logger.info(f"    Attempting to delete batch of {len(deletion_batch_ids)} leads from Instantly...")
                        for batch_lead_id in deletion_batch_ids:
                            delete_response = instantly_service.delete_lead(batch_lead_id)
                            if delete_response and (delete_response.status_code == 200 or delete_response.status_code == 204):
                                total_leads_successfully_deleted += 1
                                logger.info(f"      Successfully deleted lead {batch_lead_id} from Instantly.")
                            else:
                                total_leads_failed_to_delete += 1
                                error_text = delete_response.text if delete_response else "No response"
                                status_code = delete_response.status_code if delete_response else "N/A"
                                logger.error(f"      Failed to delete lead {batch_lead_id} from Instantly. Status: {status_code}, Response: {error_text}")
                        time.sleep(DELETION_BATCH_DELAY_SECONDS) # Wait after a batch
                    else:
                        logger.info(f"    [DRY RUN] Would delete batch of {len(deletion_batch_ids)} leads: {deletion_batch_ids}")
                        total_leads_successfully_deleted += len(deletion_batch_ids) # Simulate success for dry run count
                    deletion_batch_ids = [] # Reset batch
            
            # Process any remaining leads in the last batch
            if deletion_batch_ids:
                if not dry_run:
                    logger.info(f"    Attempting to delete final batch of {len(deletion_batch_ids)} leads from Instantly...")
                    for batch_lead_id in deletion_batch_ids:
                        delete_response = instantly_service.delete_lead(batch_lead_id)
                        if delete_response and (delete_response.status_code == 200 or delete_response.status_code == 204):
                            total_leads_successfully_deleted += 1
                            logger.info(f"      Successfully deleted lead {batch_lead_id} from Instantly.")
                        else:
                            total_leads_failed_to_delete += 1
                            error_text = delete_response.text if delete_response else "No response"
                            status_code = delete_response.status_code if delete_response else "N/A"
                            logger.error(f"      Failed to delete lead {batch_lead_id} from Instantly. Status: {status_code}, Response: {error_text}")
                else:
                    logger.info(f"    [DRY RUN] Would delete final batch of {len(deletion_batch_ids)} leads: {deletion_batch_ids}")
                    total_leads_successfully_deleted += len(deletion_batch_ids) # Simulate success for dry run count

    logger.info(f"\n--- Instantly.ai Lead Deletion Process Summary ---")
    logger.info(f"Total leads identified across all campaigns/statuses: {total_leads_identified_for_deletion}")
    logger.info(f"Total leads confirmed backed up in PostgreSQL: {total_leads_confirmed_backed_up}")
    logger.info(f"Total leads skipped (not found in backup): {total_leads_not_backed_up}")
    if dry_run:
        logger.info(f"Total leads that WOULD BE deleted from Instantly (Dry Run): {total_leads_successfully_deleted}")
    else:
        logger.info(f"Total leads SUCCESSFULLY DELETED from Instantly: {total_leads_successfully_deleted}")
    if total_leads_failed_to_delete > 0:
        logger.info(f"Total leads FAILED to delete from Instantly: {total_leads_failed_to_delete}")
    logger.info(f"--- Deletion Process Finished ---")

if __name__ == "__main__":
    # Example: Run the deletion process (Dry Run by default)
    # You can override CAMPAIGN_IDS_TO_PROCESS and LEAD_STATUSES_TO_DELETE before calling
    
    # To run for specific campaigns and statuses:
    # selected_campaigns = ["your_campaign_id_1", "your_campaign_id_2"]
    # selected_statuses = ["FILTER_LEAD_LOST", "FILTER_VAL_COMPLETED"]
    # delete_leads_from_instantly(selected_campaigns, selected_statuses, dry_run=True)
    
    # To run for the globally defined CAMPAIGN_IDS_TO_PROCESS and LEAD_STATUSES_TO_DELETE:
    # Set dry_run=False to actually delete leads.
    # WARNING: SETTING dry_run=False WILL DELETE LEADS FROM INSTANTLY.AI. USE WITH EXTREME CAUTION.
    delete_leads_from_instantly(CAMPAIGN_IDS_TO_PROCESS, LEAD_STATUSES_TO_DELETE, dry_run=False)
    # Example for actual deletion (USE CAREFULLY):
    # delete_leads_from_instantly(CAMPAIGN_IDS_TO_PROCESS, LEAD_STATUSES_TO_DELETE, dry_run=False) 