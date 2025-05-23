import requests
import os
import time
import logging
import html
import re
from datetime import datetime, timezone
from dotenv import load_dotenv

# Attempt to import AttioClient from the src directory
try:
    from src.attio_service import AttioClient
    from src.external_api_service import InstantlyAPI # Import InstantlyAPI
except ImportError as e:
    print(f"Error: Could not import required services: {e}")
    print("Please ensure src/attio_service.py and src/external_api_service.py exist and the script is run from the workspace root, or adjust import paths.")
    exit(1)

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# INSTANTLY_API_KEY and ATTIO_API_KEY are used by the respective client classes
# INSTANTLY_API_KEY = os.getenv('INSTANTLY_API_KEY') 
# ATTIO_API_KEY = os.getenv('ATTIO_ACCESS_TOKEN')

# INSTANTLY_BASE_URL = 'https://api.instantly.ai/api/v2/emails' # Now handled by InstantlyAPI class
ATTIO_PERSON_OBJECT_SLUG = 'people' # As per user's object name
ATTIO_EMAIL_THREAD_ATTRIBUTE_SLUG = 'email_thread' # Slug for the email thread attribute in Attio

# Target date for filtering emails (YYYY-MM-DDTHH:MM:SSZ)
# Emails created on or after this date will be processed
TARGET_DATE_STR = "2025-01-01T00:00:00Z"
TARGET_DATE = datetime.fromisoformat(TARGET_DATE_STR.replace('Z', '+00:00'))

PROCESSED_IDS_FILE = 'processed_instantly_to_attio_ids.txt'

# Cache for storing leads fetched from campaigns to reduce API calls
campaign_leads_cache = {}

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def load_processed_ids(filename: str) -> set:
    """Loads processed email IDs from a file."""
    processed_ids = set()
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                for line in f:
                    processed_ids.add(line.strip())
            logger.info(f"Loaded {len(processed_ids)} processed IDs from {filename}")
        except IOError as e:
            logger.error(f"Error reading {filename}: {e}. Starting with an empty set.")
    else:
        logger.info(f"{filename} not found. Starting with an empty set of processed IDs.")
    return processed_ids

def save_processed_id(filename: str, email_id: str):
    """Saves a processed email ID to a file."""
    try:
        with open(filename, 'a') as f:
            f.write(email_id + '\n')
    except IOError as e:
        logger.error(f"Error writing ID {email_id} to {filename}: {e}")

def get_name_from_email(email_address: str) -> str:
    """Generates a placeholder name from an email address."""
    if not email_address:
        return "Unknown Name"
    local_part = email_address.split('@')[0]
    name_parts = local_part.replace('.', ' ').replace('_', ' ').replace('-', ' ').split()
    return ' '.join(part.capitalize() for part in name_parts) if name_parts else "Unknown Name"

def format_email_interaction(email_data: dict) -> str:
    """Formats a single email interaction for the Attio note content."""
    subject = email_data.get('subject', 'No Subject')
    body_html_content = email_data.get('body', {}).get('html')
    body_text_content = email_data.get('body', {}).get('text')

    processed_content_for_body_variable = "" 

    if body_text_content: # Prioritize plain text
        processed_content_for_body_variable = body_text_content
    elif body_html_content: # Fallback to HTML if plain text is not available
        temp_body = body_html_content
        temp_body = re.sub(r'<script\b[^>]*>.*?</script>', '', temp_body, flags=re.IGNORECASE | re.DOTALL)
        temp_body = re.sub(r'<style\b[^>]*>.*?</style>', '', temp_body, flags=re.IGNORECASE | re.DOTALL)
        temp_body = html.unescape(temp_body)
        temp_body = re.sub(r'<[^>]+>', '', temp_body)
        processed_content_for_body_variable = re.sub(r'\r\n?', '\n', temp_body).strip()
    
    if processed_content_for_body_variable:
        lines = processed_content_for_body_variable.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped_for_quote_check = line.lstrip()
            leading_spaces = line[:len(line) - len(stripped_for_quote_check)] 
            
            if stripped_for_quote_check.startswith("> "):
                cleaned_lines.append(leading_spaces + stripped_for_quote_check[len("> "):])
            elif stripped_for_quote_check.startswith(">"):
                cleaned_lines.append(leading_spaces + stripped_for_quote_check[len(">"):])
            else:
                cleaned_lines.append(line)
        
        body = "\n".join(cleaned_lines).strip() 
        if not body: 
            body = "No body content" 
    else: 
        body = "No body content"

    if not body.strip(): 
        body = "No body content"

    raw_timestamp = email_data.get('timestamp_email')
    dt_object = None
    human_readable_date = "Date Unknown"
    original_timestamp_str = str(raw_timestamp) if raw_timestamp else "Timestamp Unknown"

    if raw_timestamp:
        try:
            dt_object = datetime.fromisoformat(raw_timestamp.replace('Z', '+00:00'))
            human_readable_date = dt_object.strftime("%B %d, %Y at %I:%M %p %Z")
            original_timestamp_str = dt_object.isoformat() 
        except ValueError as e:
            logger.warning(f"Could not parse timestamp_email '{raw_timestamp}': {e}")

    from_email_str = email_data.get('from_address_email', 'Unknown Sender')
    
    to_emails_data = email_data.get('to_address_email_list', [])
    to_emails_str = "Unknown Recipients"
    if isinstance(to_emails_data, str):
        to_emails_str = to_emails_data
    elif isinstance(to_emails_data, list):
        to_emails_str = ", ".join(to_emails_data)
    
    event_type = "Email Interaction"
    instantly_user_account = email_data.get('eaccount')

    if instantly_user_account:
        if from_email_str == instantly_user_account:
            event_type = "Email Sent"
        elif isinstance(to_emails_data, str) and to_emails_data == instantly_user_account:
            event_type = "Email Received"
        elif isinstance(to_emails_data, list) and instantly_user_account in to_emails_data:
            event_type = "Email Received"
    else:
        logger.warning(f"Missing 'eaccount' in email data ID {email_data.get('id')}, cannot accurately determine event type.")
    
    thread_entry = (
        f"Date: {human_readable_date}\n"
        f"Event Type: {event_type}\n"
        f"From: {from_email_str}\n"
        f"To: {to_emails_str}\n"
        f"Subject: {subject}\n\n"
        f"{body}\n\n"
        f"Original Timestamp: {original_timestamp_str}\n"
        f"---Instantly Email ID: {email_data.get('id', 'N/A')}---"
    )
    return thread_entry

# --- Main Logic ---
def fetch_instantly_emails(instantly_client: InstantlyAPI, target_date: datetime, max_emails_to_fetch: int = None) -> list:
    """Fetches emails from Instantly API using InstantlyAPI client, filtering by date.
    Optionally, fetching can stop once max_emails_to_fetch is reached.
    """
    filtered_emails = []
    starting_after = None
    fetched_count = 0
    page_num = 1

    logger.info(f"Starting to fetch Instantly emails created on or after {target_date.isoformat()}...")

    while True:
        try:
            logger.info(f"Fetching page {page_num} (starting_after: {starting_after or 'None'})...")
            data = instantly_client.list_emails(limit=100, starting_after=starting_after)

            if data is None:
                logger.error("Failed to fetch email batch from Instantly. Exiting fetch loop.")
                break
            
            emails_in_batch = data.get('items', [])
            if not emails_in_batch:
                logger.info("No more emails found in this batch.")
                break

            batch_fetched_count = len(emails_in_batch)
            fetched_count += batch_fetched_count
            batch_filtered_count = 0

            for email_data in emails_in_batch:
                timestamp_str = email_data.get('timestamp_created')
                if timestamp_str:
                    try:
                        email_date = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        if email_date >= target_date:
                            filtered_emails.append(email_data)
                            batch_filtered_count += 1
                            if max_emails_to_fetch and len(filtered_emails) >= max_emails_to_fetch:
                                logger.info(f"Reached max_emails_to_fetch ({max_emails_to_fetch}). Stopping fetch.")
                                break 
                    except ValueError:
                        logger.warning(f"Could not parse timestamp '{timestamp_str}' for email ID {email_data.get('id')}")
                else:
                    logger.warning(f"Missing 'timestamp_created' for email ID {email_data.get('id')}")
            
            logger.info(f"Page {page_num}: Checked {batch_fetched_count} emails, found {batch_filtered_count} matching date criteria. Total matching so far: {len(filtered_emails)}")

            if max_emails_to_fetch and len(filtered_emails) >= max_emails_to_fetch:
                break 

            starting_after = data.get('next_starting_after')
            if not starting_after:
                logger.info("Reached the end of Instantly emails.")
                break
            page_num += 1
            time.sleep(1) 

        except Exception as e:
            logger.error(f"An unexpected error occurred during Instantly email fetching loop: {e}")
            break 
            
    logger.info(f"Finished fetching Instantly emails. Total checked: {fetched_count}. Total matching date criteria: {len(filtered_emails)}.")
    return filtered_emails

def process_and_store_in_attio(emails: list, attio_client: AttioClient, instantly_client: InstantlyAPI, processed_ids: set):
    """Processes emails and stores them as notes in Attio."""
    new_persons_created_for_notes = 0
    new_notes_created = 0
    records_failed = 0
    skipped_existing_in_file = 0

    if not emails:
        logger.info("No emails to process for Attio.")
        return

    logger.info(f"Starting to process {len(emails)} emails for Attio (will create notes).")

    for email_data in emails:
        instantly_email_id = email_data.get('id')
        if not instantly_email_id:
            logger.warning("Found an Instantly email with no ID. Skipping.")
            records_failed += 1
            continue

        if instantly_email_id in processed_ids:
            skipped_existing_in_file +=1
            continue

        instantly_user_account = email_data.get('eaccount')
        from_email_str = email_data.get('from_address_email')
        to_emails_data = email_data.get('to_address_email_list', [])
        
        # Determine the primary external party email for Attio Person record
        external_party_email_for_attio = None
        if from_email_str == instantly_user_account:
            if isinstance(to_emails_data, str):
                external_party_email_for_attio = to_emails_data
            elif isinstance(to_emails_data, list) and to_emails_data:
                external_party_email_for_attio = to_emails_data[0]
        else:
            external_party_email_for_attio = from_email_str
        
        lead_email_from_instantly_obj = email_data.get('lead') # Instantly's context for this email
        if not external_party_email_for_attio:
            logger.warning(f"Instantly email ID {instantly_email_id} could not determine main external party. Using Instantly's designated lead email '{lead_email_from_instantly_obj}' if available.")
            external_party_email_for_attio = lead_email_from_instantly_obj
        
        if not external_party_email_for_attio:
             logger.error(f"Critical: Could not determine any email for Attio association for Instantly email ID {instantly_email_id}. Skipping.")
             records_failed +=1
             continue

        # Initialize names
        person_name_for_attio = None # For the Attio Person Record (contact of the email)
        campaign_client_name_for_title = "Unknown Client" # For the Note Title (campaign context)

        campaign_id = email_data.get('campaign_id')

        # Fetch/use cached campaign lead details for names
        if campaign_id and lead_email_from_instantly_obj:
            if campaign_id not in campaign_leads_cache:
                logger.info(f"Fetching leads for campaign ID {campaign_id} (for names and client info)...")
                leads_in_campaign = instantly_client.list_leads_from_campaign(campaign_id, external_party_email_for_attio)
                if leads_in_campaign is not None:
                    campaign_leads_cache[campaign_id] = {lead.get('email'): lead for lead in leads_in_campaign if lead.get('email')}
                    logger.info(f"Cached {len(campaign_leads_cache[campaign_id])} leads for campaign {campaign_id}.")
                else:
                    campaign_leads_cache[campaign_id] = {}
                    logger.warning(f"Failed to fetch or no leads found for campaign {campaign_id}.")
            
            if campaign_id in campaign_leads_cache and lead_email_from_instantly_obj in campaign_leads_cache[campaign_id]:
                detailed_lead_for_campaign_context = campaign_leads_cache[campaign_id][lead_email_from_instantly_obj]
                
                # 1. Get Client_Name for the note title from this lead's payload
                payload_data = detailed_lead_for_campaign_context.get('payload', {})
                if isinstance(payload_data, dict):
                    client_name_from_payload = payload_data.get('Client_Name')
                    if client_name_from_payload:
                        campaign_client_name_for_title = client_name_from_payload
                        logger.info(f"Using Client_Name '{campaign_client_name_for_title}' for note title from campaign lead payload.")
                    else:
                        logger.info(f"Client_Name not found in campaign lead payload for {lead_email_from_instantly_obj}. Defaulting to '{campaign_client_name_for_title}'.")
                else:
                    logger.warning(f"Campaign lead payload for {lead_email_from_instantly_obj} is not a dict: {payload_data}. Cannot get Client_Name.")

                # 2. Determine person_name_for_attio (for Attio Person, i.e., external_party_email_for_attio)
                # Only use campaign lead's name if it IS the external party of this specific email interaction
                if lead_email_from_instantly_obj == external_party_email_for_attio:
                    name_from_lead = detailed_lead_for_campaign_context.get('first_name')
                    if not name_from_lead and detailed_lead_for_campaign_context.get('full_name'):
                        name_from_lead = detailed_lead_for_campaign_context.get('full_name')
                    if name_from_lead:
                        person_name_for_attio = name_from_lead
                        logger.info(f"Using name '{person_name_for_attio}' from campaign lead ({lead_email_from_instantly_obj}) for Attio Person ({external_party_email_for_attio}).")
            else:
                 logger.info(f"Lead {lead_email_from_instantly_obj} not found in campaign {campaign_id} cache. Cannot get names/Client_Name from campaign lead data.")
        else:
            logger.info(f"Missing campaign_id or Instantly lead email ({lead_email_from_instantly_obj}) for email {instantly_email_id}. Cannot get names/Client_Name from campaign lead data.")

        # Fallback logic for person_name_for_attio if not found above (e.g. campaign lead is not the external party, or data missing)
        if not person_name_for_attio:
            current_email_from_json = email_data.get('from_address_json')
            current_email_to_json = email_data.get('to_address_json')
            is_outgoing_email = (from_email_str == instantly_user_account)

            if is_outgoing_email:
                if current_email_to_json and isinstance(current_email_to_json, list):
                    for recipient_obj in current_email_to_json:
                        if isinstance(recipient_obj, dict) and recipient_obj.get('address') == external_party_email_for_attio and recipient_obj.get('name'):
                            person_name_for_attio = recipient_obj.get('name')
                            logger.info(f"Using name '{person_name_for_attio}' from email's To JSON for Attio Person {external_party_email_for_attio}.")
                            break
            else: # Email received
                if current_email_from_json and isinstance(current_email_from_json, list) and current_email_from_json:
                     sender_obj = current_email_from_json[0]
                     if isinstance(sender_obj, dict) and sender_obj.get('address') == external_party_email_for_attio and sender_obj.get('name'):
                        person_name_for_attio = sender_obj.get('name')
                        logger.info(f"Using name '{person_name_for_attio}' from email's From JSON for Attio Person {external_party_email_for_attio}.")
            
            if not person_name_for_attio: # Final fallback for person_name_for_attio
                person_name_for_attio = get_name_from_email(external_party_email_for_attio)
                logger.info(f"Using generated name '{person_name_for_attio}' for Attio Person {external_party_email_for_attio}.")

        # Prepare for Attio Note creation
        attio_person_primary_email = external_party_email_for_attio
        formatted_email_content_for_note = format_email_interaction(email_data)
        email_subject = email_data.get('subject', 'No Subject')
        email_timestamp = email_data.get('timestamp_email')

        note_title = f"Email for {campaign_client_name_for_title} Campaign - {email_subject}"

        logger.info(f"--- Prepared for Attio Note (Instantly Email ID: {instantly_email_id}) ---")
        logger.info(f"Attio Person Target: {person_name_for_attio} ({attio_person_primary_email})")
        logger.info(f"Note Title: {note_title}")
        logger.info(f"Note Timestamp: {email_timestamp}")
        logger.info("--- End of Prepared Data for Note ---")

        # Find or Create Attio Person Record
        attio_person_record_id = None
        try:
            logger.info(f"Searching Attio for person with email: {attio_person_primary_email}")
            search_response = attio_client.filter_records(
                object_type=ATTIO_PERSON_OBJECT_SLUG,
                attribute_name="email_addresses", 
                value=[attio_person_primary_email], 
                operator="equals", 
                limit=1 
            )
            
            if search_response and search_response.get("data") and len(search_response["data"]) > 0:
                attio_person_data = search_response["data"][0]
                id_object = attio_person_data.get("id", {})
                attio_person_record_id = id_object.get("record_id")
                logger.info(f"Found existing Attio person record ID {attio_person_record_id} for email {attio_person_primary_email}.")
            else:
                logger.info(f"No existing Attio person record found for {attio_person_primary_email}. Creating a new one.")
                person_payload = {
                    "name": person_name_for_attio, 
                    "email_addresses": [attio_person_primary_email]
                }
                create_person_response = attio_client.create_record(
                    object_type=ATTIO_PERSON_OBJECT_SLUG,
                    attributes=person_payload 
                )
                created_person_id_data = create_person_response.get("data", {}).get("id", {})
                attio_person_record_id = created_person_id_data.get("record_id")
                if attio_person_record_id:
                    logger.info(f"Successfully created Attio person record {attio_person_record_id} for {attio_person_primary_email}.")
                    new_persons_created_for_notes +=1
                else:
                    logger.error(f"Failed to create or retrieve ID for new Attio person {attio_person_primary_email}. Response: {create_person_response}")
                    records_failed += 1
                    continue 
        except requests.exceptions.RequestException as re:
            logger.error(f"API Error interacting with Attio for person {attio_person_primary_email}: {re}")
            if hasattr(re, 'response') and re.response is not None: logger.error(f"Attio API Response: {re.response.text}")
            records_failed += 1
            continue 
        except Exception as e:
            logger.error(f"Unexpected error interacting with Attio for person {attio_person_primary_email}: {e}")
            records_failed += 1
            continue

        if not attio_person_record_id:
            logger.error(f"Could not find or create an Attio person record for {attio_person_primary_email}. Cannot create note.")
            records_failed +=1
            continue
            
        # Create Attio Note
        try:
            attio_client.create_note(
                parent_object_slug=ATTIO_PERSON_OBJECT_SLUG,
                parent_record_id=attio_person_record_id,
                title=note_title,
                content=formatted_email_content_for_note,
                created_at=email_timestamp,
                note_format="plaintext" 
            )
            logger.info(f"Successfully created Attio note for person {attio_person_record_id} regarding Instantly email ID {instantly_email_id}.")
            new_notes_created += 1
            
            save_processed_id(PROCESSED_IDS_FILE, instantly_email_id)
            processed_ids.add(instantly_email_id)
        
        except ValueError as ve: 
            logger.error(f"ValueError creating Attio note for Instantly email ID {instantly_email_id} (Person ID: {attio_person_record_id}): {ve}")
            records_failed += 1
        except requests.exceptions.RequestException as re:
            logger.error(f"API Error creating Attio note for Instantly email ID {instantly_email_id} (Person ID: {attio_person_record_id}): {re}")
            if hasattr(re, 'response') and re.response is not None:
                try: logger.error(f"Attio API Response Body: {re.response.json()}")
                except ValueError: logger.error(f"Attio API Response Body: {re.response.text}")
            records_failed += 1
        except Exception as e:
            logger.error(f"Unexpected error creating Attio note for Instantly email ID {instantly_email_id} (Person ID: {attio_person_record_id}): {e}")
            records_failed += 1
        
        time.sleep(0.5) 

    logger.info(f"Attio note processing summary: New persons created: {new_persons_created_for_notes}, New notes created: {new_notes_created}, Failed: {records_failed}, Skipped (already processed from file): {skipped_existing_in_file}")

if __name__ == "__main__":
    logger.info("--- Starting Instantly to Attio Sync Script ---")
    try:
        attio_client = AttioClient()
        instantly_client = InstantlyAPI()
        logger.info("AttioClient and InstantlyAPI client initialized successfully.")
    except ValueError as e: 
        logger.error(f"Failed to initialize API clients: {e}")
        exit(1)
    except Exception as e: 
        logger.error(f"An unexpected error occurred during client initialization: {e}")
        exit(1)

    processed_email_ids = load_processed_ids(PROCESSED_IDS_FILE)

    # To process all emails since TARGET_DATE_STR, ensure max_emails_to_fetch is None or not passed
    max_emails_to_process = None # Set to None for all emails

    logger.info(f"Fetching Instantly emails since {TARGET_DATE_STR} (max: {max_emails_to_process or 'all'})...")
    all_instantly_emails = fetch_instantly_emails(instantly_client, TARGET_DATE, max_emails_to_fetch=max_emails_to_process)
    
    if not all_instantly_emails:
        logger.info(f"No new emails fetched from Instantly matching the criteria. Exiting.")
    else:
        logger.info(f"--- Processing {len(all_instantly_emails)} emails for Attio ---")
        process_and_store_in_attio(all_instantly_emails, attio_client, instantly_client, processed_email_ids)

    logger.info("--- Instantly to Attio Sync Script Finished ---") 