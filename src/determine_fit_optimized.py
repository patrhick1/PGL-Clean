"""
Determine Fit Script (Optimized)

This module checks records in your "Campaign Manager" table and decides
whether a podcast is a good fit for a client. It pulls data from Airtable,
fetches existing Google Docs content, and leverages LangChain with Claude Haiku
to evaluate fit. The status of each record is then updated in Airtable.

This is an optimized version using a class-based approach with LangChain.

Author: Paschal Okonkwor
"""

import os
import json
import logging
import re
import time
import asyncio
import random
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta, timezone

# Environment and service imports
from dotenv import load_dotenv
from .airtable_service import PodcastService
from .google_docs_service import GoogleDocsService
from .data_processor import generate_prompt
from .ai_usage_tracker import tracker as ai_tracker

# LangChain components
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers.pydantic import PydanticOutputParser
from pydantic import BaseModel, Field, ValidationError

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Pydantic model for fit assessment (based on existing model in openai_service.py)
class FitAssessment(BaseModel):
    Answer: str = Field(
        description="ONLY provide your overall fit assessment by outputting either 'Fit' or 'Not a fit'",
        enum=["Fit", "Not a fit"]
    )

def sanitize_filename(name):
    """
    Remove emojis and other non-ASCII characters from a filename.
    
    Args:
        name (str): The filename to sanitize
        
    Returns:
        str: A sanitized filename
    """
    # Remove emojis and other non-ASCII characters
    name = re.sub(r'[^\w\s-]', '', name, flags=re.UNICODE)
    # Replace spaces and other unsafe characters with underscores
    name = re.sub(r'[\s/\\]', '_', name)
    # Trim leading/trailing underscores
    return name.strip('_')


class DetermineFitProcessor:
    """
    A class to process podcast records and determine if they're a good fit for a client.
    Uses LangChain with Claude to assess fit based on podcast summaries and client info.
    """
    
    def __init__(self):
        """Initialize services and LLM configuration."""
        try:
            # Initialize services
            self.airtable_service = PodcastService()
            self.google_docs_client = GoogleDocsService()
            self.parser = PydanticOutputParser(pydantic_object=FitAssessment)
            
            # LLM Configuration
            api_key = os.getenv("ANTHROPIC_API")
            if not api_key:
                raise ValueError("ANTHROPIC_API environment variable not set. Please set this in your environment or .env file.")
            """  
            self.llm = ChatAnthropic(
                model="claude-3-5-haiku-20241022",  # Using Haiku for cost efficiency
                anthropic_api_key=api_key,
                temperature=0.1
            )
            """
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", 
                google_api_key=os.getenv("GEMINI_API_KEY"), 
                temperature=0.25
            )
            # Constants (table and view names)
            self.CAMPAIGN_MANAGER_TABLE = 'Campaign Manager'
            self.CAMPAIGNS_TABLE = 'Campaigns'
            self.PODCASTS_TABLE = 'Podcasts'
            self.PODCAST_EPISODES_TABLE = 'Podcast_Episodes'
            self.OUTREACH_READY_VIEW = 'OR'
            self.PODCAST_INFO_FOLDER_ID = os.getenv('GOOGLE_PODCAST_INFO_FOLDER_ID')
            
            # Create prompt template
            self.prompt_template = self._create_prompt_template()
            
            logger.info("DetermineFitProcessor initialized successfully")
        except Exception as e:
            logger.critical(f"Failed to initialize DetermineFitProcessor: {e}", exc_info=True)
            raise
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create the prompt template for podcast fit assessment."""
        try:
            # Read the prompt template from file
            with open("prompts/prompt_determine_good_fit.txt", "r") as f:
                template = f.read()
            
            # Create the prompt template
            return PromptTemplate(
                template=template,
                input_variables=["podcast_name", "episode_summaries", "client_bio", "client_angles"]
            )
        except Exception as e:
            logger.error(f"Failed to create prompt template: {e}", exc_info=True)
            raise
    
    async def _run_llm_assessment(
        self,
        podcast_name: str,
        episode_summaries: str,
        client_bio: str,
        client_angles: str,
        podcast_id: str
    ) -> Tuple[Optional[FitAssessment], Dict[str, Any], float]:
        """
        Run the LLM to assess podcast fit for a client.
        
        Args:
            podcast_name: Name of the podcast
            episode_summaries: Text containing episode summaries
            client_bio: Client bio text
            client_angles: Client angles/topics text
            podcast_id: ID of the podcast for tracking
            
        Returns:
            Tuple of (FitAssessment object, token info dict, execution time)
        """
        token_info = {'input': 0, 'output': 0, 'total': 0}
        start_time = time.time()
        assessment_result = None
        
        try:
            formatted_prompt = self.prompt_template.format(
                podcast_name=podcast_name,
                episode_summaries=episode_summaries[:15000],  # Truncate if too long
                client_bio=client_bio,
                client_angles=client_angles
            )
            
            logger.debug(f"Formatted Prompt:\n{formatted_prompt[:500]}...")
            
            # Set up retries
            max_retries = 3
            retry_count = 0
            
            # Longer base delay for Anthropic models
            base_delay = 15 if 'anthropic' in str(self.llm).lower() else 5
            
            # Try to run the assessment with retries
            while retry_count < max_retries:
                try:
                    # Use structured output directly
                    llm_with_output = self.llm.with_structured_output(FitAssessment)
                    
                    # Execute the LLM call
                    assessment_result = await asyncio.to_thread(llm_with_output.invoke, formatted_prompt)
                    
                    # Basic validation
                    if assessment_result is None or not isinstance(assessment_result, FitAssessment):
                        raise ValueError("Invalid response structure from LLM")
                    
                    # Extract token usage
                    try:
                        input_tokens = 0
                        output_tokens = 0
                        total_tokens = 0
                        
                        # Try to extract token info from response metadata
                        try:
                            raw_response = getattr(assessment_result, '_raw_response', None) or getattr(assessment_result, 'response_metadata', None)
                            usage_metadata = getattr(raw_response, 'usage_metadata', None) if raw_response else None
                            
                            if usage_metadata:
                                input_tokens = usage_metadata.get('prompt_token_count', 0)
                                output_tokens = usage_metadata.get('candidates_token_count', 0)
                                total_tokens = usage_metadata.get('total_token_count', 0)
                        except Exception as e:
                            logger.warning(f"Error extracting token metadata: {e}")
                        
                        # Estimate if we couldn't get actual counts
                        if total_tokens == 0:
                            # Rough token estimation
                            input_tokens = len(formatted_prompt) // 4
                            output_tokens = len(str(assessment_result.model_dump())) // 4
                            total_tokens = input_tokens + output_tokens
                            
                        token_info = {
                            'input': input_tokens,
                            'output': output_tokens,
                            'total': total_tokens
                        }
                    except Exception as token_error:
                        logger.warning(f"Failed to extract token info: {token_error}")
                        # Safe defaults
                        token_info = {
                            'input': len(formatted_prompt) // 4,
                            'output': 200,
                            'total': (len(formatted_prompt) // 4) + 200
                        }
                    
                    # Success - break out of retry loop
                    break
                    
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Attempt {retry_count}/{max_retries} failed: {type(e).__name__} - {e}")
                    
                    # Check for rate limits or server errors
                    error_str = str(e).lower()
                    is_rate_limit = "quota" in error_str or "429" in error_str or "rate limit" in error_str or "concurrent" in error_str
                    is_server_error = "500" in error_str or "503" in error_str or "too many requests" in error_str
                    
                    if (is_rate_limit or is_server_error) and retry_count < max_retries:
                        # Use exponential backoff with jitter to avoid thundering herd problem
                        wait_time = base_delay * (2 ** (retry_count - 1)) + random.uniform(0, 5)
                        
                        # For rate limits specifically, add even more delay
                        if is_rate_limit:
                            wait_time *= 1.5
                            
                        logger.warning(f"Rate limit/server error. Retrying in {wait_time:.1f}s...")
                        
                        # For Anthropic rate limits, add an extra message about concurrency
                        if 'anthropic' in str(self.llm).lower() and ('concurrent' in error_str or 'rate limit' in error_str):
                            logger.warning(f"Anthropic rate limit due to concurrent connections - consider using concurrency=1 for Claude models")
                        
                        await asyncio.sleep(wait_time)
                    elif retry_count >= max_retries:
                        logger.error(f"Max retries reached. Failing assessment.")
                        raise
                    else:
                        # For non-retryable errors, apply shorter delay but still retry
                        wait_time = 2 * (retry_count)
                        logger.error(f"Non-rate-limit error. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        if retry_count >= max_retries:
                            raise
            
            execution_time = time.time() - start_time
            logger.info(f"Successfully completed fit assessment. Time: {execution_time:.2f}s, Tokens: {token_info['total']}")
            
            # Log token usage to tracker
            if assessment_result:
                model_name = getattr(self.llm, 'model', 'unknown')
                # Fix model name format by removing 'model/' prefix if present
                if isinstance(model_name, str) and '/' in model_name:
                    model_name = model_name.split('/')[-1]
                ai_tracker.log_usage(
                    workflow="determine_fit",
                    model=model_name,
                    tokens_in=token_info['input'],
                    tokens_out=token_info['output'],
                    execution_time=execution_time,
                    podcast_id=podcast_id
                )
            
            return assessment_result, token_info, execution_time
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed to complete assessment after all retries: {type(e).__name__} - {e}", exc_info=True)
            return None, token_info, execution_time
    
    async def process_single_record(self, cm_record_id: str) -> Dict[str, Any]:
        """
        Process a single Campaign Manager record to determine podcast fit.
        
        Args:
            cm_record_id: The Airtable record ID for the Campaign Manager record
            
        Returns:
            Dict containing the processing results
        """
        result = {
            'record_id': cm_record_id,
            'status': 'Error',
            'fit_assessment': None,
            'error_reason': '',
            'execution_time': 0,
            'tokens_used': 0,
            'processing_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            # Step 1: Get the Campaign Manager record
            logger.info(f"Processing Campaign Manager record with ID: {cm_record_id}")
            cm_record = self.airtable_service.get_record(self.CAMPAIGN_MANAGER_TABLE, cm_record_id)
            if not cm_record:
                result['error_reason'] = f"Failed to retrieve Campaign Manager record {cm_record_id}"
                return result
            
            cm_fields = cm_record.get('fields', {})
            
            # Step 2: Get the Campaign record for client information
            campaign_ids = cm_fields.get('Campaigns', [])
            if not campaign_ids:
                result['error_reason'] = f"No campaign linked to Campaign Manager record {cm_record_id}"
                return result
            
            campaign_id = campaign_ids[0]
            campaign_record = self.airtable_service.get_record(self.CAMPAIGNS_TABLE, campaign_id)
            if not campaign_record:
                result['error_reason'] = f"Failed to retrieve Campaign record {campaign_id}"
                return result
            
            campaign_fields = campaign_record.get('fields', {})
            bio = campaign_fields.get('TextBio', '')
            angles = campaign_fields.get('TextAngles', '')
            
            # Step 3: Get the Podcast record
            podcast_ids = cm_fields.get('Podcast Name', [])
            if not podcast_ids:
                result['error_reason'] = f"No podcast linked to Campaign Manager record {cm_record_id}"
                return result
            
            podcast_id = podcast_ids[0]
            podcast_record = self.airtable_service.get_record(self.PODCASTS_TABLE, podcast_id)
            if not podcast_record:
                result['error_reason'] = f"Failed to retrieve Podcast record {podcast_id}"
                return result
            
            podcast_fields = podcast_record.get('fields', {})
            podcast_name = podcast_fields.get('Podcast Name', '')
            sanitized_podcast_name = sanitize_filename(podcast_name)
            
            # Step 4: Get or create the Google Doc for podcast info
            google_doc_name = f"{sanitized_podcast_name} - Info"
            doc_exists, google_doc_id, created_time_str = self.google_docs_client.check_file_exists_in_folder(google_doc_name)
            
            # Check if the document is older than 14 days
            if doc_exists and created_time_str:
                try:
                    # Parse the created time string (assuming ISO 8601 format with Z for UTC)
                    created_time = datetime.fromisoformat(created_time_str.replace('Z', '+00:00'))
                    # Ensure we have timezone-aware datetime for comparison
                    now = datetime.now(timezone.utc) 
                    
                    if now - created_time > timedelta(days=14):
                        logger.info(f"Document '{google_doc_name}' (ID: {google_doc_id}) is older than 14 days. Deleting and recreating.")
                        # Delete the old document
                        deleted = self.google_docs_client.delete_file_by_id(google_doc_id)
                        if deleted:
                            doc_exists = False
                            google_doc_id = None # Reset ID
                        else:
                            logger.warning(f"Failed to delete old document {google_doc_id}. Will attempt to use existing document.")
                except Exception as date_err:
                     logger.warning(f"Could not parse created time '{created_time_str}' for doc '{google_doc_name}': {date_err}. Will use existing document.")
            
            if not doc_exists:
                # Document doesn't exist OR was just deleted, create it
                logger.info(f"Creating new Google Doc: {google_doc_name}")
                google_doc_id = self.google_docs_client.create_document_without_content(
                    google_doc_name, self.PODCAST_INFO_FOLDER_ID)
                
                # Save doc ID in Airtable
                self.airtable_service.update_record(
                    self.PODCASTS_TABLE, podcast_id, {'PodcastEpisodeInfo': google_doc_id})
                
                # Get episode IDs to add to the document
                episode_ids = podcast_fields.get('Podcast Episodes', [])
                if not episode_ids:
                    result['error_reason'] = f"No episodes linked to Podcast record {podcast_id}"
                    return result
                
                # Add episodes to the document - **OPTIMIZED**
                all_episode_content = "" # Accumulate content here
                for episode_id in episode_ids:
                    episode_record = self.airtable_service.get_record(
                        self.PODCAST_EPISODES_TABLE, episode_id)
                    if not episode_record:
                        logger.warning(f"Could not find episode record {episode_id} for podcast {podcast_id}")
                        continue # Skip this episode
                        
                    episode_fields = episode_record.get('fields', {})
                    episode_title = episode_fields.get('Episode Title', '')
                    calculation = episode_fields.get('Calculation', '')
                    summary = episode_fields.get('Summary', '')
                    ai_summary = episode_fields.get('AI Summary', '')
                    
                    episode_content = (
                        f"Episode Title: {episode_title}\n"
                        f"Episode ID: {calculation}\n"
                        f"Summary:\n{summary}\n{ai_summary}\n"
                        "End of Episode\n\n")
                    # self.google_docs_client.append_to_document(google_doc_id, episode_content) # OLD WAY
                    all_episode_content += episode_content # Add to accumulator
                
                # Make a single append call with all content
                if all_episode_content:
                    logger.info(f"Appending {len(all_episode_content)} characters of episode summaries to doc {google_doc_id}")
                    try:
                         self.google_docs_client.append_to_document(google_doc_id, all_episode_content)
                    except Exception as append_err:
                         logger.error(f"Failed to append bulk episode content to doc {google_doc_id}: {append_err}", exc_info=True)
                         # Decide how to handle: maybe raise error, maybe continue without summaries?
                         # For now, let's log the error and proceed, the LLM might still work without summaries
                         pass 
                    episode_summaries = all_episode_content # Use the accumulated content
                else:
                     logger.warning(f"No episode content found or generated for new doc {google_doc_id}")
                     episode_summaries = "" # Ensure it's an empty string
            else:
                # Document exists and is recent, get its content
                logger.info(f"Using existing Google Doc: {google_doc_name} (ID: {google_doc_id})")
                episode_summaries = self.google_docs_client.get_document_content(google_doc_id)
                
                # Update the podcast record link just in case it changed (unlikely but safe)
                self.airtable_service.update_record(
                    self.PODCASTS_TABLE, podcast_id, {'PodcastEpisodeInfo': google_doc_id})
                
                # If document is empty, add episode content (also optimized)
                if not episode_summaries.strip():
                    episode_ids = podcast_fields.get('Podcast Episodes', [])
                    if not episode_ids:
                        result['error_reason'] = f"No episodes linked to Podcast record {podcast_id} (when trying to populate empty doc)"
                        return result
                    
                    # Accumulate content first
                    all_episode_content = "" 
                    for episode_id in episode_ids:
                        episode_record = self.airtable_service.get_record(
                            self.PODCAST_EPISODES_TABLE, episode_id)
                        if not episode_record:
                           logger.warning(f"Could not find episode record {episode_id} for podcast {podcast_id} (populating empty doc)")
                           continue # Skip this episode
                           
                        episode_fields = episode_record.get('fields', {})
                        episode_title = episode_fields.get('Episode Title', '')
                        calculation = episode_fields.get('Calculation', '')
                        summary = episode_fields.get('Summary', '')
                        ai_summary = episode_fields.get('AI Summary', '')
                        
                        episode_content = (
                            f"Episode Title: {episode_title}\n"
                            f"Episode ID: {calculation}\n"
                            f"Summary:\n{summary}\n{ai_summary}\n"
                            "End of Episode\n\n")
                        # self.google_docs_client.append_to_document(google_doc_id, episode_content) # OLD WAY
                        all_episode_content += episode_content
                        
                    # Single append call for empty documents
                    if all_episode_content:
                        logger.info(f"Appending {len(all_episode_content)} characters of episode summaries to EMPTY doc {google_doc_id}")
                        try:
                            self.google_docs_client.append_to_document(google_doc_id, all_episode_content)
                        except Exception as append_err:
                            logger.error(f"Failed to append bulk episode content to EMPTY doc {google_doc_id}: {append_err}", exc_info=True)
                            pass # Log error and proceed
                        episode_summaries = all_episode_content # Use the accumulated content
                    else:
                        logger.warning(f"No episode content found or generated for empty doc {google_doc_id}")
                        episode_summaries = "" # Ensure it's an empty string
            
            # Step 5: Run the LLM assessment
            assessment, token_info, execution_time = await self._run_llm_assessment(
                podcast_name, episode_summaries, bio, angles, podcast_id
            )
            
            if assessment is None:
                result['error_reason'] = "Failed to get a valid assessment from the LLM"
                return result
            
            # Step 6: Update the status in Airtable
            fit_status = assessment.Answer
            self.airtable_service.update_record(
                self.CAMPAIGN_MANAGER_TABLE, cm_record_id, {'Status': fit_status})
            
            # Update result with success data
            result.update({
                'status': 'Success',
                'fit_assessment': fit_status,
                'execution_time': execution_time,
                'tokens_used': token_info['total']
            })
            
            logger.info(f"Successfully processed record {cm_record_id}, status: {fit_status}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing record {cm_record_id}: {e}", exc_info=True)
            result['error_reason'] = str(e)
            return result
    
    async def process_batch(self, batch_records: List[Dict], semaphore, stop_flag=None) -> List[Dict]:
        """
        Process a batch of Campaign Manager records with concurrency control.
        
        Args:
            batch_records: List of record IDs to process
            semaphore: Asyncio semaphore for concurrency control
            stop_flag: Optional event to signal when to stop processing
            
        Returns:
            List of results from processing each record
        """
        tasks = []
        request_delay = 2  # Seconds between requests
        
        for record in batch_records:
            # Check for stop flag before processing each record
            if stop_flag and stop_flag.is_set():
                logger.info("Stop flag is set - terminating during batch processing")
                break
                
            async with semaphore:
                await asyncio.sleep(request_delay)
                task = asyncio.create_task(self.process_single_record(record['id']))
                tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    async def process_all_records(self, max_concurrency=1, batch_size=1, stop_flag=None) -> Dict[str, Any]:
        """
        Process all Campaign Manager records in the 'OR' view.
        
        Args:
            max_concurrency: Maximum number of concurrent processes
            batch_size: Number of records per batch
            stop_flag: Optional event to signal when to stop processing
            
        Returns:
            Statistics about the processing
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'fit_count': 0,
            'not_fit_count': 0,
            'total_tokens': 0,
            'start_time': time.time(),
            'end_time': None,
            'duration_seconds': 0,
            'stopped_early': False
        }
        
        try:
            # Check for stop flag at the beginning
            if stop_flag and stop_flag.is_set():
                logger.info("Stop flag is set - terminating before starting processing")
                stats['stopped_early'] = True
                stats['end_time'] = time.time()
                stats['duration_seconds'] = stats['end_time'] - stats['start_time']
                return stats
                
            # Fetch records from the 'OR' view
            logger.info(f"Fetching records from the '{self.OUTREACH_READY_VIEW}' view")
            records = self.airtable_service.get_records_from_view(
                self.CAMPAIGN_MANAGER_TABLE, self.OUTREACH_READY_VIEW)
            logger.info(f"Found {len(records)} record(s) to process")
            
            # Check for stop flag after fetching records
            if stop_flag and stop_flag.is_set():
                logger.info("Stop flag is set - terminating after fetching records")
                stats['stopped_early'] = True
                stats['end_time'] = time.time()
                stats['duration_seconds'] = stats['end_time'] - stats['start_time']
                return stats
            
            if not records:
                logger.info("No records found to process")
                stats['end_time'] = time.time()
                stats['duration_seconds'] = stats['end_time'] - stats['start_time']
                return stats
            
            # Process records in batches
            batches = [records[i:i + batch_size] for i in range(0, len(records), batch_size)]
            logger.info(f"Processing {len(records)} records in {len(batches)} batches")
            
            all_results = []
            semaphore = asyncio.Semaphore(max_concurrency)
            
            for i, batch in enumerate(batches):
                # Check for stop flag before processing each batch
                if stop_flag and stop_flag.is_set():
                    logger.info(f"Stop flag is set - terminating before batch {i+1}/{len(batches)}")
                    stats['stopped_early'] = True
                    break
                    
                batch_num = i + 1
                logger.info(f"Starting Batch {batch_num}/{len(batches)} ({len(batch)} records)")
                
                if i > 0:
                    # Add a delay between batches
                    logger.info(f"Pausing for 5 seconds before batch {batch_num}...")
                    
                    # Break down the pause into 1-second intervals to check stop flag
                    for _ in range(5):
                        if stop_flag and stop_flag.is_set():
                            logger.info(f"Stop flag is set - terminating during pause before batch {batch_num}")
                            stats['stopped_early'] = True
                            break
                        await asyncio.sleep(1)
                    
                    # Check if we should terminate after the pause
                    if stop_flag and stop_flag.is_set():
                        break
                
                start_batch_time = time.time()
                batch_results = await self.process_batch(batch, semaphore, stop_flag)
                batch_duration = time.time() - start_batch_time
                
                logger.info(f"Finished Batch {batch_num}/{len(batches)}. Duration: {batch_duration:.2f}s")
                all_results.extend(batch_results)
                
                # Update stats
                for result in batch_results:
                    stats['total_processed'] += 1
                    
                    if result['status'] == 'Success':
                        stats['successful'] += 1
                        if result['fit_assessment'] == 'Fit':
                            stats['fit_count'] += 1
                        else:
                            stats['not_fit_count'] += 1
                        stats['total_tokens'] += result['tokens_used']
                    else:
                        stats['failed'] += 1
                
                # Pause after every batch
                if batch_num < len(batches):
                    pause_duration = 30
                    logger.info(f"PAUSING for {pause_duration} seconds after processing batch {batch_num}...")
                    
                    # Break down the 30-second pause into smaller intervals to check stop flag
                    for _ in range(30):
                        if stop_flag and stop_flag.is_set():
                            logger.info(f"Stop flag is set - terminating during pause after batch {batch_num}")
                            stats['stopped_early'] = True
                            break
                        await asyncio.sleep(1)
                    
                    # Check if we should terminate after the pause
                    if stop_flag and stop_flag.is_set():
                        break
            
            # Update final stats
            stats['end_time'] = time.time()
            stats['duration_seconds'] = stats['end_time'] - stats['start_time']
            
            # Log statistics
            logger.info("--- Processing Statistics ---")
            logger.info(f"  Total records processed: {stats['total_processed']}")
            logger.info(f"  Successful: {stats['successful']} ({stats['successful']/max(stats['total_processed'], 1)*100:.1f}%)")
            logger.info(f"  Failed: {stats['failed']} ({stats['failed']/max(stats['total_processed'], 1)*100:.1f}%)")
            logger.info(f"  Fit: {stats['fit_count']} ({stats['fit_count']/max(stats['successful'], 1)*100:.1f}%)")
            logger.info(f"  Not Fit: {stats['not_fit_count']} ({stats['not_fit_count']/max(stats['successful'], 1)*100:.1f}%)")
            logger.info(f"  Total tokens used: {stats['total_tokens']}")
            logger.info(f"  Average tokens per record: {stats['total_tokens']/max(stats['successful'], 1):.1f}")
            logger.info(f"  Total processing duration: {stats['duration_seconds']:.2f} seconds")
            logger.info(f"  Stopped early: {stats['stopped_early']}")
            logger.info("-----------------------------")
            
            # Save stats to file
            stats_file = f"determine_fit_stats_{timestamp}.json"
            try:
                stats_save = stats.copy()
                stats_save['start_time'] = datetime.fromtimestamp(stats_save['start_time']).isoformat()
                stats_save['end_time'] = datetime.fromtimestamp(stats_save['end_time']).isoformat()
                with open(stats_file, 'w') as f:
                    json.dump(stats_save, f, indent=2)
                logger.info(f"Processing statistics saved to {stats_file}")
            except Exception as e:
                logger.error(f"Failed to save statistics to JSON: {e}")
            
            return stats
            
        except Exception as e:
            logger.critical(f"Critical error in process_all_records: {e}", exc_info=True)
            stats['end_time'] = time.time()
            stats['duration_seconds'] = stats['end_time'] - stats['start_time']
            return stats


# Function to run for standard processing (with stop flag support)
async def determine_fit_async(stop_flag: Optional[Any] = None, max_concurrency: int = 3, batch_size: int = 5) -> Dict[str, Any]:
    """
    Async entry point for determine_fit script.
    
    Args:
        stop_flag: Optional event to signal when to stop processing
        max_concurrency: Maximum number of concurrent processes to run
        batch_size: Number of records to process in each batch
        
    Returns:
        Dictionary with processing statistics
    """
    logger.info("Starting Determine Fit Automation (Optimized)")
    logger.info(f"Using max_concurrency={max_concurrency}, batch_size={batch_size}")
    
    try:
        processor = DetermineFitProcessor()
        
        # Check if should stop before starting
        if stop_flag and stop_flag.is_set():
            logger.info("Stop flag set before starting processing")
            return {'status': 'stopped', 'message': 'Processing stopped by stop flag', 'stopped_early': True}
        
        # Process all records with explicit concurrency and batch size
        stats = await processor.process_all_records(max_concurrency=max_concurrency, batch_size=batch_size, stop_flag=stop_flag)
        
        return stats
    except Exception as e:
        logger.critical(f"Determine Fit automation failed: {e}", exc_info=True)
        return {'status': 'error', 'message': str(e)}


# Synchronous wrapper for compatibility with existing code
def determine_fit(stop_flag: Optional[Any] = None, max_concurrency: int = 3, batch_size: int = 5) -> Dict[str, Any]:
    """
    Synchronous wrapper for determine_fit_async.
    
    Args:
        stop_flag: Optional event to signal when to stop processing
        max_concurrency: Maximum number of concurrent processes to run
        batch_size: Number of records to process in each batch
        
    Returns:
        Dictionary with processing statistics
    """
    return asyncio.run(determine_fit_async(stop_flag, max_concurrency, batch_size))


# Direct execution entry point
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Configuration for direct script execution
    MAX_CONCURRENCY = 3
    BATCH_SIZE = 5
    
    logger.info("=============================================")
    logger.info("Starting Determine Fit Process (Optimized)")
    logger.info(f"Using max_concurrency={MAX_CONCURRENCY}, batch_size={BATCH_SIZE}")
    logger.info("=============================================")
    
    start_run_time = time.time()
    results = asyncio.run(determine_fit_async(max_concurrency=MAX_CONCURRENCY, batch_size=BATCH_SIZE))
    end_run_time = time.time()
    
    total_run_duration = end_run_time - start_run_time
    logger.info("=============================================")
    logger.info("Determine Fit Process Ended")
    logger.info(f"Total script execution time: {total_run_duration:.2f} seconds")
    logger.info("=============================================") 