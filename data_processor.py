# app/data_processor.py

import json 
import logging

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


class DataProcessor:
    def process_podcast_result(self, podcast_data, campaign_id, campaign_name, airtable_service):
        podcast_title = podcast_data.get('title_original')
        podcast_email = podcast_data.get('email')

        # Search for existing podcast in Airtable
        formula = f"{{Podcast Name}}=\"{podcast_title}\""
        existing_records = airtable_service.search_records('Podcasts', formula)

        if not existing_records:
            # Create a new podcast record
            podcast_record = {
                'Podcast Name': podcast_title,
                'Email': podcast_email,
                'Host Name': podcast_data.get('publisher_original'),
                'RSS Feed': podcast_data.get('rss'),
                'Status': 'New Show',
                'Active Campaign': campaign_name
            }
            new_podcast = airtable_service.create_record('Podcasts', podcast_record)
            podcast_id = new_podcast['id']

            # Create a new campaign manager record
            campaign_manager_record = {
                'Status': 'Prospect',
                'Podcast Name': [podcast_id],
                'Campaigns': [campaign_id],
                'Podcast': podcast_title
            }
            airtable_service.create_record('Campaign Manager', campaign_manager_record)
        else:
            # If the podcast exists, check if a campaign manager record exists
            podcast_id = existing_records[0]['id']
            formula = f"AND({{Podcast Name}} = \"{podcast_title}\", {{CampaignName}} = \"{campaign_name}\")"
            campaign_records = airtable_service.search_records('Campaign Manager', formula)
            if not campaign_records:
                # Create a new campaign manager record
                campaign_manager_record = {
                    'Status': 'Prospect',
                    'Podcast Name': [podcast_id],
                    'Campaigns': [campaign_id],
                    'Podcast': podcast_title
                }
                airtable_service.create_record('Campaign Manager', campaign_manager_record)
            # Else, do nothing as the campaign manager record already exists


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
            logger.error(f"Failed to parse JSON. The response was:\n{variable}")
            return None  # or return variable, depending on your use case

    # If variable is neither string nor dict, return it as-is
    return variable

