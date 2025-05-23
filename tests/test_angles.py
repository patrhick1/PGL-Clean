"""
Test script for angles.py
"""

import sys
import os

# Add the project root directory (PGL) to sys.path
# This allows imports from the 'src' directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import logging
import traceback
import asyncio
from src.airtable_service import MIPRService
from src.angles import AnglesProcessor
from src.google_docs_service import GoogleDocsService
from src.data_processor import extract_document_id

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_record_processing(record_id):
    """Test processing a record using real document content"""
    logger.info(f"Starting test for record {record_id}")
    
    try:
        # First enable the Angles & Bio Button
        airtable = MIPRService()
        logger.info(f"Enabling 'Angles & Bio Button' for record {record_id}")
        airtable.update_record(record_id, {"Angles & Bio Button": True})
        logger.info("Button enabled, now processing the record")
        
        # Get the record to check its fields
        record = airtable.get_record_by_id(record_id)
        if not record:
            logger.error(f"Could not find record {record_id}")
            return
        
        # Initialize the AnglesProcessor
        processor = AnglesProcessor()
        
        # Extract the document links and names from the record
        fields = record.get('fields', {})
        name = fields.get('Name', 'Unknown')
        
        # Get document links
        social_media_posts_link = fields.get('Social Media posts', '')
        podcast_transcripts_link = fields.get('Podcast transcripts', '')
        articles_link = fields.get('Articles', '')
        
        # Extract document IDs
        social_doc_id = extract_document_id(social_media_posts_link)
        podcast_doc_id = extract_document_id(podcast_transcripts_link)
        articles_doc_id = extract_document_id(articles_link)
        
        logger.info(f"Document IDs for {name}: Social={social_doc_id}, Podcast={podcast_doc_id}, Articles={articles_doc_id}")
        
        # Retrieve content from Google Docs
        google_docs_service = GoogleDocsService()
        logger.info("Retrieving document content...")
        
        # Get content from Google Docs
        social_content = ""
        podcast_content = ""
        articles_content = ""
        
        try:
            if social_doc_id:
                social_content = google_docs_service.get_document_content(social_doc_id)
                logger.info(f"Retrieved social content: {len(social_content)} characters")
        except Exception as e:
            logger.error(f"Error retrieving social content: {e}")
            
        try:
            if podcast_doc_id:
                podcast_content = google_docs_service.get_document_content(podcast_doc_id)
                logger.info(f"Retrieved podcast content: {len(podcast_content)} characters")
        except Exception as e:
            logger.error(f"Error retrieving podcast content: {e}")
            
        try:
            if articles_doc_id:
                articles_content = google_docs_service.get_document_content(articles_doc_id)
                logger.info(f"Retrieved articles content: {len(articles_content)} characters")
        except Exception as e:
            logger.error(f"Error retrieving articles content: {e}")
            
        # Check if we have content from at least one source
        if not (social_content or podcast_content or articles_content):
            logger.error("Could not retrieve any content from documents")
            return
            
        # Prepare content tuple (social, podcast, articles)
        content = (social_content, podcast_content, articles_content)
        
        # Get the prompt document
        logger.info("Retrieving prompt template...")
        prompt_doc_id = processor.prompt_doc_id_v1
        prompt_template = google_docs_service.get_document_content(prompt_doc_id)
        
        if not prompt_template:
            logger.error("Failed to get prompt template")
            return
        
        logger.info(f"Prompt template retrieved: {len(prompt_template)} characters")
        
        # Generate structured data
        logger.info(f"Generating structured data for {name}...")
        structured_data = await processor.generate_structured_data(
            name=name,
            content=content,
            prompt_template=prompt_template,
            version="v1"
        )
        
        # Log the result
        logger.info(f"Structured data generated successfully:")
        if 'Bio' in structured_data:
            bio_preview = structured_data['Bio'][:100] + "..." if len(structured_data['Bio']) > 100 else structured_data['Bio']
            logger.info(f"Bio preview: {bio_preview}")
        
        if 'Angles' in structured_data:
            angles_preview = structured_data['Angles'][:100] + "..." if len(structured_data['Angles']) > 100 else structured_data['Angles']
            logger.info(f"Angles preview: {angles_preview}")
        
        return structured_data
    
    except Exception as e:
        logger.error(f"Error processing record: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def main():
    """Run a test on a specific record."""
    # The record ID to test
    record_id = 'recSk7KFHjwKikNU1'  # Casey Cheshire record ID
    
    try:
        # Run the async test
        result = asyncio.run(test_record_processing(record_id))
        
        # Print the result
        if result:
            logger.info(f"Processing completed successfully")
            if 'error' in result:
                logger.error(f"Error in processing: {result['error']}")
        else:
            logger.warning("No result returned from processing")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 