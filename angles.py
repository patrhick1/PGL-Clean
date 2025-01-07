"""
Angles Generation

This module creates angles (topics and descriptions) for a client's bio in different
versions (v1, v2) based on information stored in Google Docs and Airtable. 
It uses external services like OpenAI and Anthropic to analyze content.

Author: Paschal Okonkwor
Date: 2025-01-06
"""

import logging

from airtable_service import MIPRService
from google_docs_service import GoogleDocsService
from openai_service import OpenAIService
from anthropic_service import AnthropicService
from data_processor import extract_document_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# We'll initialize the MIPRService globally so multiple functions can use it.
airtable_service = MIPRService()

def generate_angles_and_bio_v1(record_id, airtable_service):
    """
    Generate angles and a bio (version 1) for a given record in Airtable. This uses
    Google Docs content, Anthropic for text generation, and OpenAI for parsing
    the generated text into structured data.
    
    Args:
        record_id (str): The unique identifier of the Airtable record.
        airtable_service (MIPRService): An instance of the MIPRService class used to interact with Airtable.
    """
    logger.info(f"Generating Angles and Bio v1 for record: {record_id}")

    try:
        # Instantiate helper services
        google_docs_service = GoogleDocsService()
        openai_service = OpenAIService()
        anthropic_service = AnthropicService()

        # Retrieve the record from Airtable
        record = airtable_service.get_record_by_id(record_id)
        fields = record.get('fields', {})

        # Check if 'Angles & Bio Button' is true
        if not fields.get('Angles & Bio Button'):
            logger.info(f"Record {record_id} does not have 'Angles & Bio Button' enabled.")
            return

        # Get relevant fields from the record
        name = fields.get('Name', '')
        social_media_posts_link = fields.get('Social Media posts', '')
        podcast_transcripts_link = fields.get('Podcast transcripts', '')
        articles_link = fields.get('Articles', '')

        # Extract document IDs from the links
        social_doc_id = extract_document_id(social_media_posts_link)
        podcast_doc_id = extract_document_id(podcast_transcripts_link)
        articles_doc_id = extract_document_id(articles_link)

        # Fetch content from Google Docs if IDs exist
        social_content = google_docs_service.get_document_content(social_doc_id) if social_doc_id else ''
        podcast_content = google_docs_service.get_document_content(podcast_doc_id) if podcast_doc_id else ''
        articles_content = google_docs_service.get_document_content(articles_doc_id) if articles_doc_id else ''

        # Prompt document IDs
        prompt_doc_id = '1r8FUzNCWkJRdBpe87diiP645X1uOC6GJKdyGEK6s_Qs'  
        keyword_prompt_doc_id = '18r8jTqj5cCzhnlajjKTPCJ4roG7kxTTNgCGAlW5WxoM'

        # Get prompt text from Google Docs
        prompt_content = google_docs_service.get_document_content(prompt_doc_id)
        keyword_prompt_content = google_docs_service.get_document_content(keyword_prompt_doc_id)

        # Prepare the prompt for Anthropic
        prompt = f"""
        Client Name: {name}

        Social Media posts:
        {social_content}

        Podcast transcripts:
        {podcast_content}

        Articles:
        {articles_content}

        {prompt_content}

        Create the bios and angles for {name}
        """

        # Call Anthropic to generate the text
        anthropic_response = anthropic_service.create_message(prompt)

        # Use OpenAI GPT to structure the text
        parsing_prompt = "Parse out each of 3 bios and client angles, each angle has 3 parts: Topic, Outcome, Description"
        structured_data = openai_service.transform_text_to_structured_data(parsing_prompt, anthropic_response)

        # Create new Google Docs for Bio and Angles
        bio_doc_link = google_docs_service.create_document(f'{name} - Bio v1', structured_data.get("Bio"))
        bio_doc_id = bio_doc_link.split('/')[-2]
        google_docs_service.share_document(bio_doc_id)

        angles_doc_link = google_docs_service.create_document(f'{name} - Angles v1', structured_data.get("Angles"))
        angles_doc_id = angles_doc_link.split('/')[-2]
        google_docs_service.share_document(angles_doc_id)

        # Generate keywords using an additional GPT prompt
        system_prompt = keyword_prompt_content
        prompt_for_keywords = f"""
            Bio: 
            {structured_data.get("Bio")}

            Angles:
            {structured_data.get("Angles")}
        """
        keywords = openai_service.create_chat_completion(system_prompt, prompt_for_keywords)

        # Update the Airtable record
        update_fields = {
            'Bio v1': bio_doc_link,
            'Angles v1': angles_doc_link,
            'Angles & Bio Button': False,  # Reset the button so we don't generate over and over
            'Keywords': keywords
        }
        airtable_service.update_record(record_id, update_fields)

        logger.info(f"Successfully generated bios and angles for record {record_id} (v1).")

    except Exception as e:
        logger.error(f"Error generating Angles and Bio v1 for record {record_id}: {e}")

def generate_angles_and_bio_v2(record_id, airtable_service):
    """
    Generate angles and a bio (version 2) for a given record in Airtable. This checks 
    if 'Mock Interview Email Send' is also true before proceeding. It then uses 
    Google Docs, Anthropic, and OpenAI to produce and structure the text.

    Args:
        record_id (str): The unique identifier of the Airtable record.
        airtable_service (MIPRService): An instance of the MIPRService class used to interact with Airtable.
    """
    logger.info(f"Generating Angles and Bio v2 for record: {record_id}")

    try:
        # Instantiate helper services
        google_docs_service = GoogleDocsService()
        openai_service = OpenAIService()
        anthropic_service = AnthropicService()

        # Retrieve the record from Airtable
        record = airtable_service.get_record_by_id(record_id)
        fields = record.get('fields', {})

        # Check if 'Angles & Bio Button' and 'Mock Interview Email Send' are true
        has_bio_button = fields.get('Angles & Bio Button')
        has_mock_send = fields.get('Mock Interview Email Send')
        if not (has_bio_button and has_mock_send):
            logger.info(f"Record {record_id} is missing one or both required fields for v2.")
            return

        # Extract relevant fields
        name = fields.get('Name', '')
        bio_link = fields.get('Bio v1', '')
        angles_link = fields.get('Angles v1', '')
        transcript_link = fields.get('Transcription with client', '')

        # Extract document IDs
        bio_doc_id = extract_document_id(bio_link)
        angles_doc_id = extract_document_id(angles_link)
        transcript_doc_id = extract_document_id(transcript_link)

        # Fetch content from Google Docs if IDs exist
        bio_content = google_docs_service.get_document_content(bio_doc_id) if bio_doc_id else ''
        angles_content = google_docs_service.get_document_content(angles_doc_id) if angles_doc_id else ''
        transcript_content = google_docs_service.get_document_content(transcript_doc_id) if transcript_doc_id else ''

        # Prompt document for v2
        prompt_doc_id = '1hk3sietKNY29wrq9_O5iLJ1Any_lmQ8FSNY6mAY5OG8'
        prompt_content = google_docs_service.get_document_content(prompt_doc_id)

        # Prepare Anthropic prompt
        prompt = f"""
        Client Name: {name}

        Social Media posts:
        {bio_content}

        Podcast transcripts:
        {angles_content}

        Articles:
        {transcript_content}

        {prompt_content}

        Create the bios and angles for {name}
        """

        # Generate text with Anthropic
        anthropic_response = anthropic_service.create_message(prompt)

        # Structure the text with OpenAI
        parsing_prompt = "Parse out each of 3 bios and client angles, each angle has 3 parts: Topic, Outcome, Description"
        structured_data = openai_service.transform_text_to_structured_data(parsing_prompt, anthropic_response)

        # Create new Google Docs for Bio and Angles (v2)
        bio_doc_link = google_docs_service.create_document(f'{name} - Bio v2', structured_data.get("Bio"))
        bio_doc_id = bio_doc_link.split('/')[-2]
        google_docs_service.share_document(bio_doc_id)

        angles_doc_link = google_docs_service.create_document(f'{name} - Angles v2', structured_data.get("Angles"))
        angles_doc_id = angles_doc_link.split('/')[-2]
        google_docs_service.share_document(angles_doc_id)

        # Update Airtable record
        update_fields = {
            'Bio v2': bio_doc_link,
            'Angles v2': angles_doc_link,
            'Angles & Bio Button': False  # Reset the button
        }
        airtable_service.update_record(record_id, update_fields)

        logger.info(f"Successfully generated bios and angles for record {record_id} (v2).")

    except Exception as e:
        logger.error(f"Error generating Angles and Bio v2 for record {record_id}: {e}")

def filter_by_transcription_availability(record_id, airtable_service):
    """
    Check if the record has a non-empty 'Transcription with client' field.
    If it does, generate angles and bio (v2). Otherwise, generate angles and bio (v1).
    
    Args:
        record_id (str): The unique identifier of the Airtable record.
        airtable_service (MIPRService): An instance of the MIPRService class used to interact with Airtable.
    """
    try:
        logger.info(f"Filtering record {record_id} by transcription availability...")

        record = airtable_service.get_record_by_id(record_id)
        fields = record.get('fields', {})

        # Check if 'Angles & Bio Button' is true
        if not fields.get('Angles & Bio Button'):
            logger.info(f"Record {record_id} does not have 'Angles & Bio Button' enabled.")
            return

        transcription = fields.get('Transcription with client', '')
        if transcription.strip():
            # If transcription is not empty, generate v2
            generate_angles_and_bio_v2(record_id, airtable_service)
        else:
            # If transcription is empty, generate v1
            generate_angles_and_bio_v1(record_id, airtable_service)

    except Exception as e:
        logger.error(f"Error in filter_by_transcription_availability for record {record_id}: {e}")
