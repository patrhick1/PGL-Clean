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
import asyncio
import json
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
import threading
import re

# Import Replit DB for persistence
try:
    from replit import db
    REPLIT_DB_AVAILABLE = True
except ImportError:
    REPLIT_DB_AVAILABLE = False
    logging.warning("Replit DB not available, falling back to file storage")

from .airtable_service import MIPRService, PodcastService
from .openai_service import OpenAIService
from .mipr_podcast import process_mipr_podcast_search_listennotes, process_mipr_podcast_search_with_podscan
from .angles import filter_by_transcription_availability
from .gemini_search import query_gemini_with_grounding

# Configure a logger for this module
logger = logging.getLogger(__name__)

# Key for storing last processed time in Replit DB
LAST_PROCESSED_TIME_KEY = "pgl_angles_last_processed_time"

# NEW HELPER FUNCTION for name normalization
def normalize_name_comparison(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    # Lowercase
    name = name.lower()
    # Remove common titles
    name = re.sub(r'\b(mr|ms|mrs|dr|prof|rev)\b\.?\s*', '', name, flags=re.IGNORECASE)
    # Remove extra whitespace and punctuation that's not part of a name
    name = re.sub(r'[^\w\s,-]', '', name) # Keep alphanumeric, whitespace, comma, hyphen
    name = ' '.join(name.split()) # Normalize whitespace
    
    # Handle multiple names: sort them if separated by common delimiters
    # This helps if order is different but names are the same
    parts = re.split(r'\s*&\s*|\s*and\s*|\s*,\s*(?![^()]*\))', name) # Split by '&', 'and', or ', ' (not inside parentheses)
    normalized_parts = sorted([p.strip() for p in parts if p.strip()])
    return ', '.join(normalized_parts) if normalized_parts else None


def read_last_processed_time():
    """
    Read the last processed time from Replit DB or fallback to a text file.
    If not found, return a default past date.
    """
    # Default timestamp to return if no stored value is found
    default_timestamp = '2025-01-05T00:00:00.000Z'
    
    if REPLIT_DB_AVAILABLE:
        try:
            # Try to get the timestamp from Replit DB
            return db.get(LAST_PROCESSED_TIME_KEY, default_timestamp)
        except Exception as e:
            logger.error(f"Error reading from Replit DB: {e}")
            # Fall back to file if DB access fails
    
    # Fallback to file-based storage
    if os.path.exists('last_processed_time.txt'):
        try:
            with open('last_processed_time.txt', 'r') as file:
                last_time = file.read().strip()
                if last_time:
                    return last_time
        except Exception as e:
            logger.error(f"Error reading from file: {e}")
    
    # Return default if both methods fail or no value is stored
    return default_timestamp


def write_last_processed_time(last_time):
    """
    Write the last processed time to Replit DB or fallback to a text file.
    """
    if REPLIT_DB_AVAILABLE:
        try:
            # Store the timestamp in Replit DB
            db[LAST_PROCESSED_TIME_KEY] = last_time
            logger.info(f"Updated last processed time in Replit DB: {last_time}")
            return
        except Exception as e:
            logger.error(f"Error writing to Replit DB: {e}")
            # Fall back to file if DB access fails
    
    # Fallback to file-based storage
    try:
        with open('last_processed_time.txt', 'w') as file:
            file.write(last_time)
        logger.info(f"Updated last processed time in file: {last_time}")
    except Exception as e:
        logger.error(f"Error writing to file: {e}")


def poll_airtable_and_process(record_id: Optional[str] = None, stop_flag: Optional[threading.Event] = None):
    """
    Poll Airtable and process records, with the ability to stop gracefully.
    
    Args:
        record_id: Optional Airtable record ID to process a specific record
        stop_flag: Optional threading.Event that signals when to stop processing
    """
    try:
        airtable_service = MIPRService()
        
        # Check if we should stop
        if stop_flag and stop_flag.is_set():
            logger.info("Stopping poll_airtable_and_process due to stop flag")
            return
        
        # If a specific record ID is provided, process just that record
        if record_id:
            logger.info(f"Processing specific record: {record_id}")
            try:
                # Process the specific record
                result = filter_by_transcription_availability(record_id, airtable_service)
                logger.info(f"Processed record {record_id} with result: {result}")
            except Exception as e:
                logger.error(f"Error processing record {record_id}: {e}")
            return
        
        # If no record ID provided, use the original batch processing logic
        # 1. Read the last processed time
        last_processed_time = read_last_processed_time()

        # 2. Build the Airtable formula
        formula = (
            f"AND("
            f"{{Angles & Bio Button}} = TRUE(), "
            f"IS_AFTER(LAST_MODIFIED_TIME({{Angles & Bio Button}}), '{last_processed_time}')"
            f")")

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
                filter_by_transcription_availability(record_id, airtable_service)

                # Check the record's last modified time to update newest_time if needed
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
            
    except Exception as e:
        logger.error(f"Error in poll_airtable_and_process: {e}")
        raise


def poll_podcast_search_database(record_id, stop_flag: Optional[threading.Event] = None):
    """
    Poll the PodcastService for records in the 'Campaigns' table with '{Run API} = TRUE()'.
    For each record, run 'process_mipr_podcast_search' to handle any further logic. 
    If there's an error, it will be logged and the function will continue processing the rest.
    
    Args:
        record_id: The Airtable record ID to process
        stop_flag: Optional threading.Event that signals when to stop processing
    """
    try:
        # Check if we should stop
        if stop_flag and stop_flag.is_set():
            logger.info(f"Stopping podcast search for record {record_id} due to stop flag")
            return
            
        try:
            # Pass the stop_flag down
            process_mipr_podcast_search_listennotes(record_id, stop_flag=stop_flag)

        except Exception as e:
            logger.error(f"Error processing record {record_id}: {e}")
    except Exception as e:
        logger.error(f"General error in poll_podcast_search_database: {e}")


async def process_single_podcast(podcast: Dict, openai_service: OpenAIService) -> Dict[str, Any]:
    """
    Process a single podcast record to enrich the host name.
    
    Args:
        podcast: The podcast record from Airtable
        openai_service: OpenAI service instance for API calls
        
    Returns:
        Dict with processing results and statistics
    """
    start_time = time.time()
    record_id = podcast['id']
    
    result = {
        'record_id': record_id,
        'success': False,
        'error_reason': '',
        'original_host': None,
        'enriched_host': None,
        'execution_time': 0,
        'tokens_used': {'input': 0, 'output': 0}
    }
    
    try:
        fields = podcast.get('fields', {})
        
        # Extract the host rollup, handling cases where it might not exist
        host_rollup_list = fields.get('Host Rollup (from Podcast Episodes)', [])
        if not host_rollup_list:
            result['error_reason'] = "No Host Rollup available"
            return result
            
        host_rollup = host_rollup_list[0]
        result['original_host'] = host_rollup
        
        # Get email and description, handling cases where they might not exist
        email = fields.get('Email', '')
        podcast_description = fields.get('Description', '')
        podcast_name = fields.get('Podcast Name', '')
        query = f"Who is the host of the podcast: \"{podcast_name}\", with the description: {podcast_description[:100]}?\n"

        resp_text, queries, chunks = query_gemini_with_grounding(query)
        
        # UPDATED raw_text_for_analysis and prompt_for_openai
        raw_text_for_analysis = (
            f"Contextual Information Provided:\n"
            f"1. Podcast Name: \"{podcast_name}\"\n"
            f"2. Potential Host (from episode rollups, if available): \"{host_rollup if host_rollup else 'Not available'}\"\n"
            f"3. Podcast Contact Email (if available): \"{email if email else 'Not available'}\"\n"
            f"4. Podcast Description Snippet: \"{podcast_description[:1000]}\"\n"
            f"5. Google Search Results for host query: \"{resp_text[:1000]}\"\n\n" # Limit length

            f"Task: Critically analyze ALL the provided contextual information to identify the primary host(s) of the podcast.\n"
            f"Instructions for your response:\n"
            f"- Host: Identify the host(s). If multiple, separate with a comma. If uncertain or no host clearly identified, return null or empty string for 'Host'.\n"
            f"- confidence: Provide a confidence score (0.0 to 1.0). Assign high confidence (e.g., >=0.9) ONLY if there is strong, unambiguous, and corroborating evidence. For example, if the 'Host Rollup' matches a name strongly indicated as host in 'Google Search Results' or 'Podcast Description'. If evidence is weak, conflicting, or relies on a single, uncorroborated source, assign a lower confidence score.\n"
            f"- evidence_source: Specify the PRIMARY source from the 'Contextual Information Provided' (e.g., 'Host Rollup', 'Google Search Results', 'Podcast Description and Email') that most strongly supports your host identification.\n"
            f"- evidence_text: Quote the EXACT text snippets or specific reasoning from the context that led to your identification, confidence, and source. Explain how you weighed different pieces of information. If 'Host Rollup' was used, state if it matched other findings.\n"
            f"- discrepancies_found: If you find conflicting information between sources (e.g., Host Rollup suggests 'Alice' but Google Search suggests 'Bob' is the host), describe this discrepancy. If no major discrepancies, state 'None'.\n"
            f"Prioritize accuracy and be conservative in your judgment. It's better to have lower confidence than to be wrong."
        )

        prompt_for_openai = ("Based *only* on the contextual information I've provided, identify the podcast host(s), your confidence, the primary evidence source, the supporting evidence text, and any discrepancies found. "
                             "Follow the structured output format precisely.")
        
        structured_data = openai_service.transform_text_to_structured_data(
            prompt=prompt_for_openai,
            raw_text=raw_text_for_analysis,
            data_type='host_name_analysis', 
            workflow='enrich_host_name_v3', # New version
            podcast_id=record_id
        )
        
        # Extract the host name, confidence, and evidence
        host_name = structured_data.get('Host')
        confidence = structured_data.get('confidence')
        evidence_source = structured_data.get('evidence_source')
        evidence_text = structured_data.get('evidence_text', '')
        discrepancies = structured_data.get('discrepancies_found')

        result['enriched_host'] = host_name
        result['confidence'] = confidence
        result['evidence_source'] = evidence_source
        result['evidence_text'] = evidence_text
        result['discrepancies'] = discrepancies
        
        # UPDATED HostConfirmed Logic
        host_confirmed = False
        confirmation_reason = "Initial state: Not confirmed."

        if host_name and confidence is not None: 
            normalized_openai_host = normalize_name_comparison(host_name)
            normalized_rollup_host = normalize_name_comparison(host_rollup) if host_rollup else None

            if confidence >= 0.90 and normalized_rollup_host and normalized_openai_host == normalized_rollup_host:
                if "Host Rollup" in (evidence_source or ""):
                    host_confirmed = True
                    confirmation_reason = f"Strong: OpenAI Conf ({confidence:.2f}) matches Host Rollup ('{host_rollup}'). Evidence source: {evidence_source}."
                else:
                    host_confirmed = True
                    confirmation_reason = f"Strong (Implicit Rollup Match): OpenAI Conf ({confidence:.2f}) matches Host Rollup ('{host_rollup}'). OpenAI cited: {evidence_source}."
            
            if not host_confirmed and confidence >= 0.95:
                strong_sources = ["Google Search Results", "Podcast Description"]
                if any(src in (evidence_source or "") for src in strong_sources):
                    if normalized_rollup_host and normalized_openai_host != normalized_rollup_host and not discrepancies:
                        confirmation_reason = f"High OpenAI Conf ({confidence:.2f}) but potential conflict with Host Rollup ('{host_rollup}') not addressed in discrepancies. OpenAI cited: {evidence_source}."
                        logger.warning(f"Record {record_id}: Host '{host_name}' (Conf: {confidence:.2f}) from '{evidence_source}' differs from Rollup '{host_rollup}' but no discrepancy noted by AI. Not confirming.")
                    elif discrepancies and "None" not in (discrepancies or "None"):
                        confirmation_reason = f"High OpenAI Conf ({confidence:.2f}) but AI noted discrepancies: '{discrepancies}'. OpenAI cited: {evidence_source}."
                        logger.warning(f"Record {record_id}: Host '{host_name}' (Conf: {confidence:.2f}) has noted discrepancies: '{discrepancies}'. Not confirming.")
                    else:
                        host_confirmed = True
                        confirmation_reason = f"Strong: OpenAI Very High Conf ({confidence:.2f}) from strong source ('{evidence_source}'). No major conflict with Rollup if present."
                else:
                    confirmation_reason = f"High OpenAI Conf ({confidence:.2f}) but evidence source ('{evidence_source}') not considered strong enough on its own, or Rollup conflict."

            if not host_confirmed and confidence >= 0.75 and normalized_rollup_host and normalized_openai_host == normalized_rollup_host:
                if "Host Rollup" in (evidence_source or ""):
                    host_confirmed = True
                    confirmation_reason = f"Moderate: OpenAI Conf ({confidence:.2f}) but perfect Host Rollup ('{host_rollup}') match & cited as source."
                else:
                    confirmation_reason = f"Moderate OpenAI Conf ({confidence:.2f}), matches Rollup, but Rollup not primary AI source. AI cited: {evidence_source}."

            if not host_confirmed and host_name:
                if confidence < 0.75:
                    confirmation_reason = f"Low OpenAI Confidence ({confidence:.2f})."
                elif normalized_rollup_host and normalized_openai_host != normalized_rollup_host:
                     confirmation_reason += f" Mismatch with Host Rollup ('{host_rollup}' vs OpenAI: '{host_name}')."
                if not confirmation_reason or confirmation_reason == "Initial state: Not confirmed.":
                    confirmation_reason = f"OpenAI Conf ({confidence:.2f}). Host Rollup: '{host_rollup}'. OpenAI Host: '{host_name}'. Evidence: '{evidence_source}'. No rule met for confirmation."

        elif not host_name:
            confirmation_reason = "No host name returned by OpenAI."
        
        result['host_confirmed_by_logic'] = host_confirmed
        result['confirmation_reason'] = confirmation_reason

        if host_confirmed:
            logger.info(f"Host identification for {record_id} CONFIRMED as '{host_name}'. Reason: {confirmation_reason}")
        else:
            logger.info(f"Host identification for {record_id} (Host: '{host_name}', Conf: {confidence}, Source: {evidence_source}) NOT CONFIRMED. Reason: {confirmation_reason}")
        
        # Set token usage - this would ideally come from the API response
        result['tokens_used'] = {
            'input': len(raw_text_for_analysis) // 4,  # rough estimate
            'output': len(host_name) // 4  # rough estimate
        }
        
        # Check if host name is valid
        if not host_name:
            result['error_reason'] = "Empty host name returned from OpenAI"
            return result
            
        # Update result with success
        result['success'] = True
        result['execution_time'] = time.time() - start_time
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing podcast record {record_id}: {str(e)}")
        result['error_reason'] = str(e)
        result['execution_time'] = time.time() - start_time
        return result


async def process_podcasts_batch(podcasts: List[Dict], openai_service: OpenAIService, 
                                airtable_service: PodcastService,
                                semaphore, update_airtable: bool = True) -> List[Dict]:
    """
    Process a batch of podcast records with concurrency control.
    
    Args:
        podcasts: List of podcast records
        openai_service: OpenAI service instance
        airtable_service: Airtable service instance
        semaphore: Asyncio semaphore for concurrency control
        update_airtable: Whether to update Airtable with the results
        
    Returns:
        List of processing results
    """
    async def process_with_semaphore(podcast):
        async with semaphore:
            result = await process_single_podcast(podcast, openai_service)
            
            # Update Airtable if requested and processing was successful
            if update_airtable and result['success']:
                try:
                    airtable_update_data = {'Host Name': result['enriched_host']}
                    
                    # Use the new confirmation logic's output
                    host_confirmed_status = result.get('host_confirmed_by_logic', False)
                    airtable_update_data['HostConfirmed'] = host_confirmed_status
                    
                    # Optionally, store the reason and AI's confidence for review in Airtable
                    if 'confirmation_reason' in result:
                       airtable_update_data['Host Confirmation Reason'] = result['confirmation_reason'][:990] # Truncate
                    if 'confidence' in result and result['confidence'] is not None:
                       airtable_update_data['AI Host Confidence'] = result['confidence']
                    if 'evidence_source' in result:
                        airtable_update_data['AI Evidence Source'] = result['evidence_source']
                    if 'discrepancies' in result and result['discrepancies']:
                        airtable_update_data['AI Discrepancies'] = result['discrepancies']

                    airtable_service.update_record(
                        'Podcasts', 
                        result['record_id'], 
                        airtable_update_data
                    )
                    logger.info(f"Updated host name for {result['record_id']} to '{result['enriched_host']}' (HostConfirmed: {host_confirmed_status}). Reason: {result.get('confirmation_reason')}")
                except Exception as e:
                    logger.error(f"Failed to update Airtable record {result['record_id']}: {e}")
                    result['error_reason'] = f"Airtable update failed: {e}"
                    result['success'] = False
            
            return result
    
    # Create tasks for all podcasts in the batch
    tasks = [process_with_semaphore(podcast) for podcast in podcasts]
    
    # Execute all tasks and return results
    return await asyncio.gather(*tasks)


async def enrich_host_name_async(stop_flag: Optional[threading.Event] = None, 
                               batch_size: int = 5, 
                               max_concurrency: int = 3,
                               update_airtable: bool = True) -> Dict[str, Any]:
    """
    Async function to enrich host names in the Podcasts table.
    
    Args:
        stop_flag: Optional threading.Event to signal when to stop processing
        batch_size: Size of batches for processing
        max_concurrency: Maximum number of concurrent API calls
        update_airtable: Whether to update Airtable with the results
        
    Returns:
        Dictionary with processing statistics
    """
    start_time = time.time()
    stats = {
        'total_processed': 0,
        'successful': 0,
        'failed': 0,
        'total_tokens': {'input': 0, 'output': 0},
        'total_execution_time': 0,
        'results': []
    }
    
    try:
        # Initialize services
        airtable_service = PodcastService()
        openai_service = OpenAIService()

                
        # Fetch podcast records from the 'Host rollup' view
        logger.info("Fetching podcast records from 'Host rollup' view")
        podcast_records = airtable_service.search_records('Podcasts', view='Host rollup')
        
        if not podcast_records:
            logger.info("No podcast records found in 'Host rollup' view")
            return {**stats, 'end_time': time.time(), 'duration': time.time() - start_time}
        
        logger.info(f"Found {len(podcast_records)} podcast records to process")
        
        # Create batches for processing
        batches = [podcast_records[i:i + batch_size] for i in range(0, len(podcast_records), batch_size)]
        logger.info(f"Divided into {len(batches)} batches of size {batch_size}")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrency)
        
        # Process batches
        for i, batch in enumerate(batches):
            # Check if we should stop
            if stop_flag and stop_flag.is_set():
                logger.info("Stopping host name enrichment due to stop flag")
                break
                
            batch_num = i + 1
            logger.info(f"Processing batch {batch_num}/{len(batches)} ({len(batch)} records)")
            
            # Add a small delay between batches to avoid rate limiting
            if i > 0:
                await asyncio.sleep(2)
            
            # Process the batch
            batch_results = await process_podcasts_batch(
                batch, openai_service, airtable_service, semaphore, update_airtable
            )
            
            # Update statistics
            for result in batch_results:
                stats['total_processed'] += 1
                stats['results'].append(result)
                
                if result['success']:
                    stats['successful'] += 1
                    stats['total_tokens']['input'] += result['tokens_used']['input']
                    stats['total_tokens']['output'] += result['tokens_used']['output']
                else:
                    stats['failed'] += 1
            
            logger.info(f"Completed batch {batch_num}. "
                       f"Success: {sum(1 for r in batch_results if r['success'])}/{len(batch_results)}")
        
        # Calculate final statistics
        stats['total_execution_time'] = time.time() - start_time
        
        # Log summary
        logger.info("=== Host Name Enrichment Summary ===")
        logger.info(f"Total processed: {stats['total_processed']}")
        logger.info(f"Successful: {stats['successful']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Total tokens - Input: {stats['total_tokens']['input']}, Output: {stats['total_tokens']['output']}")
        logger.info(f"Total execution time: {stats['total_execution_time']:.2f} seconds")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error in enrich_host_name_async: {e}", exc_info=True)
        stats['error'] = str(e)
        stats['total_execution_time'] = time.time() - start_time
        return stats


def enrich_host_name(stop_flag: Optional[threading.Event] = None):
    """
    Enrich the host name using an asynchronous approach with batching and concurrency.
    
    Args:
        stop_flag: Optional threading.Event that signals when to stop processing
    """
    try:
        logger.info("Starting host name enrichment with optimized implementation")
        
        # Create and start a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async enrichment function
            results = loop.run_until_complete(
                enrich_host_name_async(
                    stop_flag=stop_flag,
                    batch_size=5,        # Process 5 podcasts per batch
                    max_concurrency=3,   # Allow 3 concurrent API calls
                    update_airtable=True # Update Airtable with the results
                )
            )
            
            # Log completion
            logger.info(f"Host name enrichment completed. Processed {results['total_processed']} records.")
            logger.info(f"Success rate: {results['successful']/max(results['total_processed'], 1)*100:.1f}%")
            
        finally:
            # Clean up the loop
            loop.close()
            
    except Exception as e:
        logger.error(f"Error in enrich_host_name: {e}", exc_info=True)
        raise