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
import os
from airtable_service import MIPRService, PodcastService
from openai_service import OpenAIService
from mipr_podcast import process_mipr_podcast_search_listennotes, process_mipr_podcast_search_with_podscan
from angles import filter_by_transcription_availability
from typing import Optional
import threading

# Configure a logger for this module
logger = logging.getLogger(__name__)


def read_last_processed_time():
    """
    Read the last processed time from 'last_processed_time.txt'
    If not found, return a default past date.
    """
    if os.path.exists('last_processed_time.txt'):
        with open('last_processed_time.txt', 'r') as file:
            last_time = file.read().strip()
            if last_time:
                return last_time
    # If the file doesn't exist or is empty, return a default
    return '2025-01-05T00:00:00.000Z'  # or any date/time in ISO format


def write_last_processed_time(last_time):
    """
    Write the last processed time to 'last_processed_time.txt'
    """

    with open('last_processed_time.txt', 'w') as file:
        file.write(last_time)


def poll_airtable_and_process(stop_flag: Optional[threading.Event] = None):
    """
    Poll Airtable and process records, with the ability to stop gracefully.
    
    Args:
        stop_flag: Optional threading.Event that signals when to stop processing
    """
    try:
        while True:
            # Check if we should stop
            if stop_flag and stop_flag.is_set():
                logger.info("Stopping poll_airtable_and_process due to stop flag")
                break
                
            # 1. Read the last processed time
            last_processed_time = read_last_processed_time()

            # 2. Build the Airtable formula
            #    use LAST_MODIFIED_TIME({Angles & Bio Button}).
            formula = (
                f"AND("
                f"{{Angles & Bio Button}} = TRUE(), "
                f"IS_AFTER(LAST_MODIFIED_TIME({{Angles & Bio Button}}), '{last_processed_time}')"
                f")")

            airtable_service = MIPRService()

            # 3. Fetch records from Airtable matching the condition
            records = airtable_service.get_records_with_filter(formula)
            logger.info(f"Found {len(records)} new/modified records to process.")

            # We'll keep track of the "latest" modification time among processed records
            newest_time = last_processed_time

            for record in records:
                # Check stop flag periodically
                if stop_flag and stop_flag.is_set():
                    logger.info("Stopping record processing due to stop flag")
                    return
                    
                record_id = record['id']
                try:
                    # Run our logic on each record
                    filter_by_transcription_availability(record_id,
                                                         airtable_service)

                    # Check the record's last modified time to update newest_time if needed
                    # returns data. Suppose we assume it has record['fields']['Last Modified']:
                    record_modified_time = record['fields'].get('Claude Generate')

                    # Compare and update our newest_time
                    if record_modified_time and record_modified_time > newest_time:
                        newest_time = record_modified_time

                except Exception as e:
                    logger.error(f"Error processing record {record_id}: {e}")

            # 4. After processing all records, write the newest_time back to file,
            #    so next time we only get changes that happened after this run.
            if len(records) > 0:
                write_last_processed_time(newest_time)

            # If this is meant to run once and exit, break here
            break
            
    except Exception as e:
        logger.error(f"Error in poll_airtable_and_process: {e}")
        raise


def poll_podcast_search_database(record_id):
    """
    Poll the PodcastService for records in the 'Campaigns' table with '{Run API} = TRUE()'.
    For each record, run 'process_mipr_podcast_search' to handle any further logic. 
    If there's an error, it will be logged and the function will continue processing the rest.
    """
    try:
        try:
            #process_mipr_podcast_search_listennotes(record_id)
            process_mipr_podcast_search_with_podscan(record_id)
        except Exception as e:
            logger.error(f"Error processing record {record_id}: {e}")
    except Exception as e:
        logger.error(f"General error in poll_podcast_search_database: {e}")

def enrich_host_name():
    """
    Enrich the host name.
    """
    airtable_service = PodcastService()
    openai_service = OpenAIService()
    
    podcast_records = airtable_service.search_records('Podcasts', view='Host rollup')

    for podcast in podcast_records:
        record_id = podcast['id']
        fields = podcast['fields']
        host_rollup = fields.get('Host Rollup (from Podcast Episodes)')[0]
        email = fields.get('Email')

        # Example of transforming raw text into structured data
        raw_text = f"The Host rollup is {host_rollup}  \nand the email is {email}."
        structured_data = openai_service.transform_text_to_structured_data(
            prompt="Determine who the exact host is from this list, return only the host name",
            raw_text=raw_text,
            data_type='host_name',
            workflow='enrich_host_name',
            podcast_id=record_id)
        host_name = structured_data.get('Host')
        airtable_service.update_record('Podcasts', record_id, {'Host Name': host_name})

