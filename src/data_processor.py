# app/data_processor.py

import json
from email.utils import parsedate_to_datetime
from datetime import datetime, timedelta
import logging
import html
from typing import Optional
import threading

logger = logging.getLogger(__name__)


def extract_document_id(google_doc_link):
    """
    Extracts the Google Document ID from a given link.
    """
    try:
        parts = google_doc_link.split('/')
        index = parts.index('d')
        document_id = parts[index + 1]
        return document_id
    except (ValueError, IndexError):
        return None


def parse_date(date_string):
    """
    Parse various date string formats into datetime objects.
    Handles timezone information correctly.
    Returns None if parsing fails.
    """
    if not date_string:
        return None

    # Handle milliseconds (common in ListenNotes)
    if isinstance(date_string, int) and len(str(date_string)) == 13: # Check if it looks like a Unix timestamp in ms
        try:
            # Convert milliseconds to seconds
            timestamp_sec = date_string / 1000
            dt_object = datetime.fromtimestamp(timestamp_sec) # Assumes local timezone, adjust if UTC needed
            # To make it timezone-aware (e.g., UTC):
            # from datetime import timezone
            # dt_object = datetime.fromtimestamp(timestamp_sec, tz=timezone.utc)
            return dt_object
        except ValueError:
            logger.warning(f"Could not parse timestamp integer: {date_string}")
            return None

    # Standard string formats
    formats = [
        "%Y-%m-%dT%H:%M:%SZ",        # ISO 8601 UTC (Zulu)
        "%Y-%m-%dT%H:%M:%S%z",       # ISO 8601 with timezone offset
        "%Y-%m-%dT%H:%M:%S.%f%z",    # ISO 8601 with timezone offset and microseconds
        "%a, %d %b %Y %H:%M:%S %Z",  # RFC 5322 (e.g., Tue, 10 Oct 2023 14:30:00 GMT)
        "%a, %d %b %Y %H:%M:%S %z",  # RFC 5322 with numeric timezone offset
        "%Y-%m-%d",                  # Simple date
        # Add other formats as needed
    ]

    for fmt in formats:
        try:
            # Attempt to parse with the current format
            dt_object = datetime.strptime(str(date_string), fmt)

            # If the format includes %z, the object is already timezone-aware
            # If it's a naive object (like from %Y-%m-%d), you might want to assign a default timezone
            # if dt_object.tzinfo is None:
            #     from datetime import timezone
            #     dt_object = dt_object.replace(tzinfo=timezone.utc) # Example: Assign UTC

            return dt_object
        except ValueError:
            continue # Try the next format
        except TypeError:
             logger.warning(f"Type error parsing date string: {date_string} (type: {type(date_string)})")
             return None # Cannot parse this type

    logger.warning(f"Could not parse date string with any known format: {date_string}")
    return None


# Helper function to manage Campaign Manager records
def _ensure_campaign_manager_record(podcast_id, podcast_name, campaign_id, campaign_name, airtable_service):
    """
    Checks if a Campaign Manager record exists for the given podcast and campaign.
    If not, creates one.
    """
    try:
        # Escape single quotes in names for the formula
        safe_podcast_name = podcast_name.replace("'", "\'")
        safe_campaign_name = campaign_name.replace("'", "\'")

        # Search using Podcast text field and Campaigns lookup field
        cm_formula = f"AND({{Podcast}} = '{safe_podcast_name}', {{Campaigns}} = '{safe_campaign_name}')"

        # Previous formula using FIND/ARRAYJOIN:
        # cm_formula = f"AND(FIND('{podcast_id}', ARRAYJOIN({{Podcast Name}})), FIND('{campaign_id}', ARRAYJOIN({{Campaigns}})))"
        logger.info(f"Searching Campaign Manager with formula: {cm_formula}")

        existing_cm_records = airtable_service.search_records('Campaign Manager', formula=cm_formula)

        if not existing_cm_records:
            logger.info(f"No existing Campaign Manager record found for Podcast '{podcast_name}' and Campaign '{campaign_name}'. Creating one.")
            campaign_manager_record = {
                'Status': 'Prospect',
                'Podcast Name': [podcast_id], # Link to the podcast record
                'Campaigns': [campaign_id], # Link to the campaign record
                'Podcast': podcast_name    # Store podcast title text for display/reference
                # NOTE: We are not explicitly setting 'Campaign Name' here,
                # assuming it's a lookup field derived from 'Campaigns'
            }
            # Create the Campaign Manager record
            cm_record = airtable_service.create_record('Campaign Manager', campaign_manager_record)
            if cm_record:
                logger.info(f"Successfully created Campaign Manager record {cm_record.get('id')} for {podcast_name}.")
            else:
                logger.error(f"Failed to create Campaign Manager record for podcast {podcast_name} (ID: {podcast_id}).")
        else:
            logger.info(f"Campaign Manager record already exists for Podcast '{podcast_name}' and Campaign '{campaign_name}'.")

    except Exception as cm_e:
        logger.error(f"Error checking/creating Campaign Manager record for podcast {podcast_id}: {cm_e}")


class DataProcessor:

    def process_podcast_result_with_listennotes(self, result, campaign_id, campaign_name, airtable_service, podscan_podcast_id=None, stop_flag: Optional[threading.Event] = None):
        """
        Processes a single podcast result from ListenNotes search and updates Airtable.
        """
        # Extract data from the result
        podcast_name = html.unescape(result.get('title_original', 'N/A'))
        podcast_url = result.get('website', 'N/A')
        podcast_description = html.unescape(result.get('description_original', 'N/A'))
        email = result.get('email')
        rss_url = result.get('rss')
        publisher = result.get('publisher_original')
        total_episodes = result.get('total_episodes')
        earliest_pub_date_ms = result.get('earliest_pub_date_ms')
        latest_pub_date_ms = result.get('latest_pub_date_ms')

        # Basic validation
        if not email or not podcast_name or podcast_name == 'N/A':
            logger.warning(f"Skipping result due to missing email or title: {result.get('id')}")
            return

        # ---------------------------------------------------------------
        # NEW LOGIC: Skip processing if this podcast is already linked to
        # the current campaign in the Campaign Manager table.
        # ---------------------------------------------------------------
        try:
            # Escape single quotes for Airtable formula
            safe_podcast_name = podcast_name.replace("'", "\'")
            safe_campaign_name = campaign_name.replace("'", "\'")

            cm_formula = f"AND({{Podcast}} = '{safe_podcast_name}', {{CampaignName}} = '{safe_campaign_name}')"
            logger.info(f"Checking Campaign Manager for existing link with formula: {cm_formula}")

            existing_cm_records = airtable_service.search_records('Campaign Manager', formula=cm_formula)

            if existing_cm_records:
                logger.info(
                    f"Campaign Manager already contains Podcast '{podcast_name}' for Campaign '{campaign_name}'. Skipping update/create for this result."
                )
                return  # Skip any further processing for this podcast result

        except Exception as cm_check_err:
            # If there's an error with the CM lookup, log it but continue processing to be safe.
            logger.error(
                f"Error checking Campaign Manager linkage for Podcast '{podcast_name}' and Campaign '{campaign_name}': {cm_check_err}"
            )

        # Parse dates if available
        latest_pub_date_dt = parse_date(latest_pub_date_ms)
        earliest_pub_date_dt = parse_date(earliest_pub_date_ms)

        # Prepare the dictionary for Airtable update, matching actual Airtable field names
        field_to_update = {
            "Podcast Name": podcast_name,
            "Description": podcast_description, # Correct Airtable field name
            "Email": email,
            "RSS Feed": rss_url,
            "Fetched": False, # Default to False, will be updated by episode fetcher
            "PodcastEpisodeInfo": " "
        }

        # Add Last Posted date if successfully parsed
        if latest_pub_date_dt:
            field_to_update["Last Posted"] = latest_pub_date_dt.strftime('%Y-%m-%d')

        # Add Podscan ID if found
        if podscan_podcast_id:
            field_to_update["Podcast id"] = podscan_podcast_id
            # "Source": "ListenNotes/Podscan" # Removed as per user edit

        # Check if the record already exists by RSS Feed URL using filterByFormula
        logger.info(f"Searching for existing podcast with RSS Feed: {rss_url}")
        formula = f"{{RSS Feed}} = '{rss_url}'"
        existing_records = airtable_service.search_records('Podcasts', formula=formula)

        if existing_records:
            record_id = existing_records[0]['id']
            record_fields = existing_records[0]['fields']
            logger.info(f"Found existing record for {podcast_name} (ID: {record_id}) with RSS Feed {rss_url}. Preparing update.")
            # Log the data being sent for update
            # logger.debug(f"Update data for {record_id}: {json.dumps(field_to_update, indent=2)}") # Optional: log full data
            try:
                campaign_record = record_fields.get('Campaign', [])
                if campaign_id not in campaign_record:
                    campaign_record.append(campaign_id)
                    field_to_update['Campaign'] = campaign_record
                    logger.info(f"Updated campaign for {podcast_name} to {campaign_record}")
                    
                # Update existing podcast record
                airtable_service.update_record('Podcasts', record_id, field_to_update)
                logger.info(f"Successfully updated record {record_id} for {podcast_name}.")

                # Ensure Campaign Manager record exists (passing campaign_name)
                _ensure_campaign_manager_record(record_id, podcast_name, campaign_id, campaign_name, airtable_service)

            except Exception as e:
                logger.error(f"Failed to update record {record_id} for {podcast_name}: {e}")
        else:
            # No existing podcast record found by RSS Feed URL, create new podcast AND campaign manager record
            logger.info(f"No existing record found for RSS Feed {rss_url}. Preparing to create new record for {podcast_name}.")
            # Add campaign link for the new record
            field_to_update['Campaign'] = [campaign_id]
            try:
                # Create the new podcast record
                field_to_update['Host Name'] = publisher
                new_record = airtable_service.create_record('Podcasts', field_to_update)
                if new_record:
                    new_podcast_id = new_record.get('id')
                    logger.info(f"Successfully created new podcast record for {podcast_name} (ID: {new_podcast_id}).")

                    # Ensure Campaign Manager record exists for the new podcast (passing campaign_name)
                    _ensure_campaign_manager_record(new_podcast_id, podcast_name, campaign_id, campaign_name, airtable_service)
                else:
                     logger.error(f"Failed to create new podcast record for {podcast_name} with RSS Feed {rss_url}, cannot create Campaign Manager record.")

            except Exception as e:
                 logger.error(f"Failed to create new podcast record for {podcast_name} with RSS Feed {rss_url}: {e}")

    def process_podcast_result_with_podscan(self, result, campaign_id, campaign_name, airtable_service, stop_flag: Optional[threading.Event] = None):
        """
        Processes a single podcast result from Podscan search and updates Airtable.
        """
        # Extract data from the result
        podcast_name = result.get('podcast_name', 'N/A')
        podcast_url = result.get('podcast_url')
        podcast_description = result.get('podcast_description')
        email = result.get('email')
        rss_url = result.get('rss_url')
        last_posted_at = result.get('last_posted_at') # Already formatted as YYYY-MM-DD
        podcast_id = result.get('podcast_id')

        # Basic validation
        if not email or not podcast_name or podcast_name == 'N/A':
            logger.warning(f"Skipping Podscan result due to missing email or title: {podcast_id}")
            return

        # Prepare the dictionary for Airtable update
        field_to_update = {
            "Podcast Name": podcast_name,
            "Podcast URL": podcast_url,
            "Podcast Description": podcast_description,
            "Email": email,
            "RSS Feed": rss_url,
            "Campaign Name": campaign_name,
            "Fetched": False, # Default to False
            "Podcast id": podcast_id, # Podscan always provides this
            "PodcastEpisodeInfo": " "
        }

        # Add dates if parsed successfully
        if last_posted_at:
            field_to_update["Last Published Date"] = last_posted_at

        # Check if the record already exists by RSS Feed URL using filterByFormula
        logger.info(f"Searching for existing Podscan podcast with RSS Feed: {rss_url}")
        formula = f"{{RSS Feed}} = '{rss_url}'"
        existing_records = airtable_service.search_records('Podcasts', formula=formula)

        if existing_records:
            record_id = existing_records[0]['id']
            record_fields = existing_records[0]['fields']
            logger.info(f"Found existing record for Podscan podcast {podcast_name} (ID: {record_id}) with RSS Feed {rss_url}. Preparing update.")
            # logger.debug(f"Update data for {record_id}: {json.dumps(field_to_update, indent=2)}") # Optional
            try:
                # Add current campaign to existing list if not present
                campaign_record = record_fields.get('Campaign', [])
                if campaign_id not in campaign_record:
                    campaign_record.append(campaign_id)
                    field_to_update['Campaign'] = campaign_record
                    logger.info(f"Updated campaign for Podscan podcast {podcast_name} to {campaign_record}")
                
                # Update the podcast record    
                airtable_service.update_record('Podcasts', record_id, field_to_update)
                logger.info(f"Successfully updated Podscan record {record_id} for {podcast_name}.")

                # Ensure Campaign Manager record exists (passing campaign_name)
                _ensure_campaign_manager_record(record_id, podcast_name, campaign_id, campaign_name, airtable_service)

            except Exception as e:
                logger.error(f"Failed to update Podscan record {record_id} for {podcast_name}: {e}")
        else:
            # No existing record found, create new podcast AND campaign manager record
            logger.info(f"No existing record found for RSS Feed {rss_url}. Preparing to create new Podscan record for {podcast_name}.")
            # logger.debug(f"Create data for {podcast_name}: {json.dumps(field_to_update, indent=2)}") # Optional
            # Add campaign link for the new record
            field_to_update['Campaign'] = [campaign_id]
            try:
                # Create the new podcast record
                new_record = airtable_service.create_record('Podcasts', field_to_update)
                if new_record:
                    new_podcast_id = new_record.get('id')
                    logger.info(f"Successfully created new Podscan record for {podcast_name} (ID: {new_podcast_id}).")
                    # Ensure Campaign Manager record exists for the new podcast (passing campaign_name)
                    _ensure_campaign_manager_record(new_podcast_id, podcast_name, campaign_id, campaign_name, airtable_service)
                else:
                    logger.error(f"Failed to create new Podscan record for {podcast_name} with RSS Feed {rss_url}, cannot create Campaign Manager record.")
            except Exception as e:
                 logger.error(f"Failed to create new Podscan record for {podcast_name} with RSS Feed {rss_url}: {e}")


def generate_prompt(placeholders, prompt_file):
    """
    Replaces placeholders in the prompt template with their respective values.

    Args:
        placeholders (dict): A dictionary where keys are placeholders and values are their replacements.
        prompt_file (str) : A string to store the path of the prompt

    Returns:
        str: The modified prompt with placeholders replaced.
    """
    with open(prompt_file, 'r') as file:
        prompt = file.read()

    for key, value in placeholders.items():
        prompt = prompt.replace(f'{{{key}}}', value)

    return prompt


def convert_into_json_format(variable):
    """
    Converts a string into JSON format if possible.
    If the variable is already a dictionary, it returns as-is.
    If the conversion fails, logs the error and returns the original variable.
    """
    if isinstance(variable, dict):
        # Already in JSON format
        return variable

    if isinstance(variable, str):
        try:
            # Attempt to parse the string into JSON
            return json.loads(variable)
        except json.JSONDecodeError:
            logger.error(
                f"Failed to parse JSON. The response was:\n{variable}")
            return None  # or return variable, depending on your use case

    # If variable is neither string nor dict, return it as-is
    return variable
