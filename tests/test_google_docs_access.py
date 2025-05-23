"""
Test script for Google Docs access permissions
"""

import logging
import os
import traceback
import sys
from google_docs_service import GoogleDocsService
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_service_account_info():
    """Get and display information about the service account being used"""
    service_account_file = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'service-account-key.json')
    
    if not os.path.exists(service_account_file):
        logger.error(f"Service account file not found: {service_account_file}")
        return
    
    try:
        import json
        with open(service_account_file, 'r') as f:
            sa_data = json.load(f)
            
        logger.info(f"Service account details:")
        logger.info(f"  Email: {sa_data.get('client_email', 'Not found')}")
        logger.info(f"  Project ID: {sa_data.get('project_id', 'Not found')}")
    except Exception as e:
        logger.error(f"Error reading service account file: {e}")

def check_document_access(doc_ids):
    """Test access to specific Google Docs documents"""
    logger.info("Initializing Google Docs service...")
    try:
        service = GoogleDocsService()
        
        # Test each document ID
        for name, doc_id in doc_ids:
            logger.info(f"\nTesting access to {name} document (ID: {doc_id})")
            
            try:
                # First try to get metadata - this requires less permissions
                logger.info("Attempting to get document metadata...")
                metadata = service.drive_service.files().get(
                    fileId=doc_id, 
                    fields="name,mimeType,owners,permissions"
                ).execute()
                logger.info(f"Successfully retrieved document metadata:")
                logger.info(f"  Name: {metadata.get('name')}")
                logger.info(f"  Type: {metadata.get('mimeType')}")
                if 'owners' in metadata:
                    logger.info(f"  Owner: {metadata['owners'][0].get('emailAddress')}")
                permissions = metadata.get('permissions', [])
                logger.info(f"  Permission count: {len(permissions)}")
                
                # Now try to get the content - this requires more permissions
                logger.info("Attempting to retrieve document content...")
                content = service.get_document_content(doc_id)
                content_preview = content[:100] + "..." if len(content) > 100 else content
                logger.info(f"SUCCESS! Retrieved {len(content)} characters")
                logger.info(f"Content preview: {content_preview}")
                
            except Exception as e:
                logger.error(f"Error accessing document: {e}")
                logger.error(traceback.format_exc())
    
    except Exception as e:
        logger.error(f"Error initializing Google Docs service: {e}")
        logger.error(traceback.format_exc())

def main():
    """Run tests for Google Docs access"""
    logger.info("Starting Google Docs access tests...")
    
    # Log environment information
    logger.info(f"Environment variables:")
    logger.info(f"  GOOGLE_APPLICATION_CREDENTIALS: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")
    logger.info(f"  GOOGLE_PODCAST_INFO_FOLDER_ID: {os.getenv('GOOGLE_PODCAST_INFO_FOLDER_ID')}")
    
    # Get service account info
    get_service_account_info()
    
    # Test access to prompt documents (known to work)
    prompt_documents = [
        ('Prompt Doc V1', '1r8FUzNCWkJRdBpe87diiP645X1uOC6GJKdyGEK6s_Qs'),
        ('Keyword Prompt', '18r8jTqj5cCzhnlajjKTPCJ4roG7kxTTNgCGAlW5WxoM'),
        ('Prompt Doc V2', '1hk3sietKNY29wrq9_O5iLJ1Any_lmQ8FSNY6mAY5OG8')
    ]
    
    # Test access to client documents (failing with permission errors)
    client_documents = [
        ('Social Media Posts', '1F3ROJiFhrrv4G5AcVg1vJmfEBnjtD1Gt_lj4O-qBbNE'),
        ('Podcast Transcripts', '1J87ma_NZDkEX3IdDPntFzFKN4Q4aRcPULDMFH802AbM'),
        ('Articles', '1gLazbEi8F0OitvxAKVt9XOyQV0KP0bESfyaR4ELaNcg')
    ]
    
    logger.info("\n=== Testing access to prompt documents ===")
    check_document_access(prompt_documents)
    
    logger.info("\n=== Testing access to client documents ===")
    check_document_access(client_documents)
    
    logger.info("\nGoogle Docs access testing complete.")

if __name__ == "__main__":
    main() 