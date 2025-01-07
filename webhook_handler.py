"""
Webhook Handler

This module defines functions that poll Airtable and run certain processes. 
It includes logic for checking records in Airtable and processing them if 
they match a certain condition (like a button being pressed). 
In this example, we poll every 5 minutes for new tasks.

Author: Paschal Okonkwor
Date: 2025-01-06
"""

import time
import logging
from airtable_service import MIPRService, PodcastService
from mipr_podcast import process_mipr_podcast_search
from angles import filter_by_transcription_availability

# Configure a logger for this module
logger = logging.getLogger(__name__)

def poll_airtable_and_process():
    """
    Poll the MIPRService for records with the field '{Angles & Bio Button} = TRUE()'.
    Then run 'filter_by_transcription_availability' on each record. If there's an error,
    it will be logged and we continue with the rest.
    """
    try:
        airtable_service = MIPRService()
        formula = "{Angles & Bio Button} = TRUE()"

        # Fetch records from Airtable that match the condition
        records = airtable_service.get_records_with_filter(formula)
        logger.info(f"Found {len(records)} records with Angles & Bio Button = TRUE.")

        for record in records:
            record_id = record['id']
            try:
                filter_by_transcription_availability(record_id, airtable_service)
            except Exception as e:
                logger.error(f"Error processing record {record_id}: {e}")

        # Wait 5 minutes before the next poll to avoid repeated processing
        time.sleep(300)
    except Exception as e:
        logger.error(f"General error polling Airtable: {e}")

def poll_podcast_search_database():
    """
    Poll the PodcastService for records in the 'Campaigns' table with '{Run API} = TRUE()'.
    For each record, run 'process_mipr_podcast_search' to handle any further logic. 
    If there's an error, it will be logged and the function will continue processing the rest.
    """
    try:
        airtable_service = PodcastService()
        # Condition for the records we want to process
        formula = "{Run API} = TRUE()"

        # Fetch matching records
        records = airtable_service.search_records('Campaigns', formula)
        logger.info(f"Found {len(records)} records with Run API = TRUE.")

        for record in records:
            record_id = record['id']
            try:
                process_mipr_podcast_search(record_id)
            except Exception as e:
                logger.error(f"Error processing record {record_id}: {e}")
    except Exception as e:
        logger.error(f"General error in poll_podcast_search_database: {e}")
