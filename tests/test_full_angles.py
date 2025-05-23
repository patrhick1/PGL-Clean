"""
Test script for the full angles.py processing workflow
"""

import logging
import sys
import traceback
import asyncio
import time
import random
from airtable_service import MIPRService
from angles import filter_by_transcription_availability

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def is_rate_limit_error(error_str):
    """Check if an error is related to rate limiting"""
    rate_limit_indicators = [
        "429", "quota", "rate limit", "too many requests", 
        "exceeded", "throttle", "resource exhausted"
    ]
    error_lower = error_str.lower()
    return any(indicator in error_lower for indicator in rate_limit_indicators)

def test_full_angles_process(record_id, max_retries=3):
    """Test the full angles processing workflow for a record with retries for rate limits"""
    logger.info(f"Starting full test for record {record_id}")
    
    retry_count = 0
    base_delay = 10  # Start with 10 seconds
    
    while retry_count <= max_retries:
        try:
            # First enable the Angles & Bio Button
            airtable = MIPRService()
            logger.info(f"Enabling 'Angles & Bio Button' for record {record_id}")
            airtable.update_record(record_id, {"Angles & Bio Button": True})
            logger.info("Button enabled, now processing the record")
            
            # Get current record data
            record = airtable.get_record_by_id(record_id)
            if not record:
                logger.error(f"Could not find record {record_id}")
                return
                
            fields = record.get('fields', {})
            name = fields.get('Name', 'Unknown')
            logger.info(f"Processing record for {name}")
            
            # Note the time for performance measurement
            start_time = time.time()
            
            # Process the record using the main function
            result = filter_by_transcription_availability(record_id)
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
            
            # Log the result
            logger.info(f"Processing result: {result}")
            
            # Check if documents were created
            if result.get('status') == 'success':
                bio_link = result.get('bio_link', '')
                angles_link = result.get('angles_link', '')
                keywords = result.get('keywords', '')
                
                logger.info(f"Bio document: {bio_link}")
                logger.info(f"Angles document: {angles_link}")
                
                if keywords:
                    logger.info(f"Keywords generated: {keywords[:100]}..." if len(keywords) > 100 else keywords)
                    
                # Check if Airtable was updated
                updated_record = airtable.get_record_by_id(record_id)
                if updated_record:
                    updated_fields = updated_record.get('fields', {})
                    bio_v1_link = updated_fields.get('Bio v1', '')
                    angles_v1_link = updated_fields.get('Angles v1', '')
                    
                    if bio_v1_link and angles_v1_link:
                        logger.info("Airtable record was successfully updated with document links")
                    else:
                        logger.warning("Airtable record was not properly updated with document links")
            
            return result
            
        except Exception as e:
            error_str = str(e)
            logger.error(f"Error processing record: {e}")
            
            # Check if this is a rate limit error
            if is_rate_limit_error(error_str) and retry_count < max_retries:
                retry_count += 1
                # Add jitter to delay (±20%)
                jitter_factor = 1 + random.uniform(-0.2, 0.2)
                retry_delay = base_delay * (2 ** retry_count) * jitter_factor
                
                logger.warning(f"Rate limit error detected. Retry {retry_count}/{max_retries} after {retry_delay:.1f}s delay")
                time.sleep(retry_delay)
                continue
            else:
                logger.error(traceback.format_exc())
                return {"status": "error", "error": error_str}

def main():
    """Run a full test on a specific record."""
    # The record ID to test
    record_id = 'recSk7KFHjwKikNU1'  # Casey Cheshire record ID
    
    try:
        # Run the test with retries
        logger.info("Starting full angles test with rate limit handling")
        result = test_full_angles_process(record_id)
        
        # Summarize the results
        if result:
            status = result.get('status', 'unknown')
            if status == 'success':
                logger.info("✅ Full processing completed successfully")
            elif status == 'skipped':
                reason = result.get('reason', 'Unknown reason')
                logger.info(f"⏭️ Processing was skipped: {reason}")
            elif status == 'error':
                error = result.get('error', 'Unknown error')
                logger.error(f"❌ Processing failed with error: {error}")
            else:
                logger.warning(f"⚠️ Processing completed with unknown status: {status}")
        else:
            logger.warning("⚠️ No result returned from processing")
        
    except Exception as e:
        logger.error(f"❌ Error in main function: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 