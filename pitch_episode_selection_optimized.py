"""
Pitch Episode Selection Script (Optimized)

This module selects the best podcast episode and angle for pitching a client 
to a specific podcast. It uses data from Airtable (Campaign Manager, Campaigns,
Podcasts, etc.), reads summarized episode information from Google Docs, and
uses LangChain with Claude Haiku to determine which episodes are best for pitching.

This is an optimized version using a class-based approach with LangChain.

Author: Paschal Okonkwor
"""

import os
import json
import logging
import time
import asyncio
import re
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

# Environment and service imports
from dotenv import load_dotenv
from airtable_service import PodcastService
from google_docs_service import GoogleDocsService
from data_processor import generate_prompt
from ai_usage_tracker import tracker as ai_tracker

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

# Pydantic models for structured outputs
class EpisodeSelection(BaseModel):
    """Episode selection response model"""
    ID: str = Field(
        description="The ID of the best episode for the client, extracted from the episode information, The ID start with rec and is 17 characters long."
    )

class PitchTopic(BaseModel):
    """Model for pitch topics and descriptions"""
    topic_1: str = Field(..., description="Title of the first topic")
    description_1: str = Field(..., description="Description for the first topic")
    topic_2: str = Field(..., description="Title of the second topic")
    description_2: str = Field(..., description="Description for the second topic")
    topic_3: str = Field(..., description="Title of the third topic")
    description_3: str = Field(..., description="Description for the third topic")


class PitchEpisodeProcessor:
    """
    A class to process podcast records and select the best episode and angles for pitching.
    Uses LangChain with Claude to assess fit based on podcast summaries and client info.
    """
    
    def __init__(self):
        """Initialize services and LLM configuration."""
        try:
            # Initialize services
            self.airtable_service = PodcastService()
            self.google_docs_client = GoogleDocsService()
            self.episode_parser = PydanticOutputParser(pydantic_object=EpisodeSelection)
            self.pitch_parser = PydanticOutputParser(pydantic_object=PitchTopic)
            
            # LLM Configuration
            api_key = os.getenv("ANTHROPIC_API")
            if not api_key:
                raise ValueError("ANTHROPIC_API environment variable not set. Please set this in your environment or .env file.")
                
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
            self.FIT_VIEW = 'Fit'
            
            # Create prompt templates
            self.episode_selection_template = self._create_episode_selection_template()
            self.pitch_writing_template = self._create_pitch_writing_template()
            
            logger.info("PitchEpisodeProcessor initialized successfully")
        except Exception as e:
            logger.critical(f"Failed to initialize PitchEpisodeProcessor: {e}", exc_info=True)
            raise
    
    def _create_episode_selection_template(self) -> PromptTemplate:
        """Create the prompt template for episode selection."""
        try:
            # Read the prompt template from file
            with open("prompts/pitch_episodes_angles_selection_prompts/prompt_claude_get_episode_id.txt", "r") as f:
                template = f.read()
                
            # Check for and fix the ID= placeholder issue
            if "{ID=}" in template or "{\\nID=\\n}" in template or "{\nID=\n}" in template:
                logger.warning("Found problematic 'ID=' placeholder in template. Fixing it.")
                template = template.replace("{ID=}", "")
                template = template.replace("{\\nID=\\n}", "")
                template = template.replace("{\nID=\n}", "")
            
            # Create the prompt template
            return PromptTemplate(
                template=template,
                input_variables=["Name (from Client)", "Podcast Name", "TextBio", "TextAngles", "text"]
            )
        except Exception as e:
            logger.error(f"Failed to create episode selection template: {e}", exc_info=True)
            # Create a fallback template if the file can't be loaded
            emergency_template = """
            You are a podcast episode selection expert.
            
            CLIENT NAME: {Name (from Client)}
            PODCAST NAME: {Podcast Name}
            CLIENT BIO: {TextBio}
            CLIENT ANGLES: {TextAngles}
            
            PODCAST EPISODE INFORMATION:
            {text}
            
            Based on the client information and podcast episodes, select the most relevant episode ID for the client.
            The episode ID starts with 'rec' and is 17 characters long.
            
            Your answer should be ONLY the episode ID.
            """
            logger.warning("Using emergency template for episode selection")
            return PromptTemplate(
                template=emergency_template,
                input_variables=["Name (from Client)", "Podcast Name", "TextBio", "TextAngles", "text"]
            )
    
    def _create_pitch_writing_template(self) -> PromptTemplate:
        """Create the prompt template for pitch writing."""
        try:
            # Read the prompt template from file
            with open("prompts/pitch_episodes_angles_selection_prompts/prompt_write_pitch.txt", "r") as f:
                template = f.read()
            
            # Create the prompt template
            return PromptTemplate(
                template=template,
                input_variables=["Name (from Client)", "Podcast Name", "Episode Title", "Summary", "AI Summary", "TextAngles"]
            )
        except Exception as e:
            logger.error(f"Failed to create pitch writing template: {e}", exc_info=True)
            raise
    
    async def _run_episode_selection(
        self,
        client_name: str,
        podcast_name: str,
        bio: str,
        angles: str,
        podcast_episode_content: str,
        podcast_id: str
    ) -> Tuple[Optional[str], Dict[str, int], float]:
        """
        Run the LLM to select the best episode with retries and error handling.
        
        Returns:
            Tuple containing:
            - The episode ID (or None if failed)
            - Token usage information
            - Execution time in seconds
        """
        episode_id = None
        token_info = {'input': 0, 'output': 0, 'total': 0}
        start_time = time.time()
        
        try:
            # Format the prompt
            try:
                formatted_prompt = self.episode_selection_template.format(
                    **{
                        "Name (from Client)": client_name,
                        "Podcast Name": podcast_name,
                        "TextBio": bio,
                        "TextAngles": angles,
                        "text": podcast_episode_content
                    }
                )
            except KeyError as e:
                logger.error(f"KeyError in prompt formatting: {e}. This suggests the prompt template contains a placeholder that wasn't provided.")
                # Try to read the template and identify the issue
                try:
                    with open("prompts/pitch_episodes_angles_selection_prompts/prompt_claude_get_episode_id.txt", "r") as f:
                        template_content = f.read()
                    logger.error(f"Template placeholders might include: {re.findall(r'{([^{}]*)}', template_content)}")
                except Exception as read_error:
                    logger.error(f"Couldn't read template file: {read_error}")
                
                # Use a simpler emergency template
                logger.warning("Using emergency template instead")
                emergency_template = """
                You are a podcast episode selection expert.
                
                CLIENT NAME: {client_name}
                PODCAST NAME: {podcast_name}
                CLIENT BIO: {client_bio}
                CLIENT ANGLES: {client_angles}
                
                PODCAST EPISODE INFORMATION:
                {podcast_content}
                
                Based on the client information and podcast episodes, select the most relevant episode ID for the client.
                The episode ID starts with 'rec' and is 17 characters long.
                
                Your answer should be ONLY the episode ID.
                """
                
                formatted_prompt = emergency_template.format(
                    client_name=client_name,
                    podcast_name=podcast_name,
                    client_bio=bio,
                    client_angles=angles,
                    podcast_content=podcast_episode_content
                )
            
            # Set up retries
            max_retries = 3
            retry_count = 0
            retry_delay = 5
            
            while retry_count < max_retries:
                try:
                    # Use the LLM with structured output
                    llm_with_output = self.llm.with_structured_output(EpisodeSelection)
                    
                    # Run the LLM call in a thread to not block the event loop
                    result = await asyncio.to_thread(llm_with_output.invoke, formatted_prompt)
                    
                    # Basic validation
                    if not result or not isinstance(result, EpisodeSelection):
                        raise ValueError("Invalid or empty response structure received from LLM.")
                    
                    # Extract usage information
                    try:
                        # Try accessing usage metadata from raw response
                        raw_response = getattr(result, '_raw_response', None) or getattr(result, 'response_metadata', None)
                        usage_metadata = getattr(raw_response, 'usage', None) if raw_response else None
                        
                        if usage_metadata:
                            input_tokens = getattr(usage_metadata, 'input_tokens', 0)
                            output_tokens = getattr(usage_metadata, 'output_tokens', 0)
                            total_tokens = input_tokens + output_tokens
                        else:
                            # Estimate if metadata not available
                            input_tokens = len(formatted_prompt) // 4
                            output_tokens = 100
                            total_tokens = input_tokens + output_tokens
                    except Exception as token_err:
                        logger.warning(f"Failed to extract token info: {token_err}")
                        # Safe defaults
                        input_tokens = len(formatted_prompt) // 4
                        output_tokens = 100
                        total_tokens = input_tokens + output_tokens
                    
                    token_info = {'input': input_tokens, 'output': output_tokens, 'total': total_tokens}
                    
                    # Log AI usage
                    execution_time = time.time() - start_time
                    ai_tracker.log_usage(
                        workflow="pitch_episode_selection",
                        model="claude-3-5-haiku-20241022",
                        tokens_in=token_info['input'],
                        tokens_out=token_info['output'],
                        execution_time=execution_time,
                        endpoint="langchain.anthropic",
                        podcast_id=podcast_id
                    )
                    
                    episode_id = result.ID
                    break  # Success, exit retry loop
                    
                except (ValidationError, ValueError) as parse_error:
                    retry_count += 1
                    logger.warning(f"Episode selection attempt {retry_count}/{max_retries} failed due to parsing/validation error: {parse_error}")
                    if retry_count >= max_retries:
                        logger.error(f"Max retries reached after parsing/validation error.")
                        raise
                    await asyncio.sleep(retry_delay)
                    
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Episode selection attempt {retry_count}/{max_retries} failed: {type(e).__name__} - {e}")
                    
                    # Check for rate limits or server errors
                    error_str = str(e).lower()
                    is_rate_limit = "quota" in error_str or "429" in error_str or "rate limit" in error_str
                    is_server_error = "500" in error_str or "503" in error_str
                    
                    if (is_rate_limit or is_server_error) and retry_count < max_retries:
                        wait_time = retry_delay * (2 ** (retry_count - 1))
                        logger.warning(f"Rate limit/server error. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    elif retry_count >= max_retries:
                        logger.error(f"Max retries reached. Failing episode selection.")
                        raise
                    else:
                        logger.error(f"Non-retryable error encountered. Failing episode selection.")
                        raise
            
            execution_time = time.time() - start_time
            logger.info(f"Successfully completed episode selection. Time: {execution_time:.2f}s, Tokens: {token_info['total']}")
            return episode_id, token_info, execution_time
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed to complete episode selection after all retries: {type(e).__name__} - {e}", exc_info=True)
            return None, token_info, execution_time
    
    async def _run_pitch_writing(
        self,
        client_name: str,
        podcast_name: str,
        episode_title: str,
        summary: str,
        ai_summary: str,
        angles: str,
        podcast_id: str
    ) -> Tuple[Optional[PitchTopic], Dict[str, int], float]:
        """
        Run the LLM to write a pitch with retries and error handling.
        
        Returns:
            Tuple containing:
            - The pitch topics (or None if failed)
            - Token usage information
            - Execution time in seconds
        """
        pitch_result = None
        token_info = {'input': 0, 'output': 0, 'total': 0}
        start_time = time.time()
        
        try:
            # Format the prompt
            formatted_prompt = self.pitch_writing_template.format(**{
                "Name (from Client)": client_name,
                "Podcast Name": podcast_name,
                "Episode Title": episode_title,
                "Summary": summary,
                "AI Summary": ai_summary,
                "TextAngles": angles
            })
            
            # Set up retries
            max_retries = 3
            retry_count = 0
            retry_delay = 5
            
            while retry_count < max_retries:
                try:
                    # Use the LLM with structured output
                    llm_with_output = self.llm.with_structured_output(PitchTopic)
                    
                    # Run the LLM call in a thread to not block the event loop
                    result = await asyncio.to_thread(llm_with_output.invoke, formatted_prompt)
                    
                    # Basic validation
                    if not result or not isinstance(result, PitchTopic):
                        raise ValueError("Invalid or empty response structure received from LLM.")
                    
                    # Extract usage information
                    try:
                        # Try accessing usage metadata from raw response
                        raw_response = getattr(result, '_raw_response', None) or getattr(result, 'response_metadata', None)
                        usage_metadata = getattr(raw_response, 'usage', None) if raw_response else None
                        
                        if usage_metadata:
                            input_tokens = getattr(usage_metadata, 'input_tokens', 0)
                            output_tokens = getattr(usage_metadata, 'output_tokens', 0)
                            total_tokens = input_tokens + output_tokens
                        else:
                            # Estimate if metadata not available
                            input_tokens = len(formatted_prompt) // 4
                            output_tokens = 250  # Pitch output is larger
                            total_tokens = input_tokens + output_tokens
                    except Exception as token_err:
                        logger.warning(f"Failed to extract token info: {token_err}")
                        # Safe defaults
                        input_tokens = len(formatted_prompt) // 4
                        output_tokens = 250
                        total_tokens = input_tokens + output_tokens
                    
                    token_info = {'input': input_tokens, 'output': output_tokens, 'total': total_tokens}
                    
                    # Log AI usage
                    execution_time = time.time() - start_time
                    ai_tracker.log_usage(
                        workflow="pitch_episode_selection",
                        model="gemini-2.0-flash",
                        tokens_in=token_info['input'],
                        tokens_out=token_info['output'],
                        execution_time=execution_time,
                        endpoint="langchain.google",
                        podcast_id=podcast_id
                    )
                    
                    pitch_result = result
                    break  # Success, exit retry loop
                    
                except (ValidationError, ValueError) as parse_error:
                    retry_count += 1
                    logger.warning(f"Pitch writing attempt {retry_count}/{max_retries} failed due to parsing/validation error: {parse_error}")
                    if retry_count >= max_retries:
                        logger.error(f"Max retries reached after parsing/validation error.")
                        raise
                    await asyncio.sleep(retry_delay)
                    
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Pitch writing attempt {retry_count}/{max_retries} failed: {type(e).__name__} - {e}")
                    
                    # Check for rate limits or server errors
                    error_str = str(e).lower()
                    is_rate_limit = "quota" in error_str or "429" in error_str or "rate limit" in error_str
                    is_server_error = "500" in error_str or "503" in error_str
                    
                    if (is_rate_limit or is_server_error) and retry_count < max_retries:
                        wait_time = retry_delay * (2 ** (retry_count - 1))
                        logger.warning(f"Rate limit/server error. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    elif retry_count >= max_retries:
                        logger.error(f"Max retries reached. Failing pitch writing.")
                        raise
                    else:
                        logger.error(f"Non-retryable error encountered. Failing pitch writing.")
                        raise
            
            execution_time = time.time() - start_time
            logger.info(f"Successfully completed pitch writing. Time: {execution_time:.2f}s, Tokens: {token_info['total']}")
            return pitch_result, token_info, execution_time
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed to complete pitch writing after all retries: {type(e).__name__} - {e}", exc_info=True)
            return None, token_info, execution_time
    
    async def process_single_record(self, cm_record_id: str) -> Dict[str, Any]:
        """
        Process a single Campaign Manager record to select the best episode and pitch angles.
        
        Args:
            cm_record_id: The Airtable record ID for the Campaign Manager record
            
        Returns:
            Dict containing the processing results
        """
        result = {
            'record_id': cm_record_id,
            'status': 'Error',
            'episode_id': None,
            'pitch_topics': None,
            'error_reason': '',
            'execution_time': 0,
            'tokens_used': 0,
            'processing_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        start_time = time.time()
        
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
            client_names = campaign_fields.get('Name (from Client)', [])
            client_name = client_names[0] if client_names else 'Unknown Client'
            
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
            podcast_episode_info = podcast_fields.get('PodcastEpisodeInfo', '')
            
            if not podcast_episode_info:
                result['error_reason'] = f"No PodcastEpisodeInfo found for podcast {podcast_id}"
                return result
            
            # Step 4: Get content from the Google Doc
            podcast_episode_content = self.google_docs_client.get_document_content(podcast_episode_info)
            
            if not podcast_episode_content.strip():
                result['error_reason'] = f"Empty podcast episode content for podcast {podcast_id}"
                return result
            
            # Step 5: Run episode selection
            selected_episode_id, episode_tokens, episode_time = await self._run_episode_selection(
                client_name, podcast_name, bio, angles, podcast_episode_content, podcast_id
            )
            
            if not selected_episode_id:
                result['error_reason'] = "Failed to select an episode"
                result['execution_time'] = time.time() - start_time
                result['tokens_used'] = episode_tokens['total']
                return result
            
            # Step 6: Get the selected episode details
            try:
                episode_record = self.airtable_service.get_record(self.PODCAST_EPISODES_TABLE, selected_episode_id)
                if not episode_record:
                    result['error_reason'] = f"Failed to retrieve selected episode record {selected_episode_id} - record not found"
                    result['episode_id'] = selected_episode_id  # Still include the episode ID
                    result['execution_time'] = time.time() - start_time
                    result['tokens_used'] = episode_tokens['total']
                    
                    # Still update the Campaign Manager record with the episode ID but without pitch topics
                    try:
                        update_fields = {
                            'Pitch Episode': selected_episode_id
                        }
                        self.airtable_service.update_record(self.CAMPAIGN_MANAGER_TABLE, cm_record_id, update_fields)
                        logger.info(f"Updated Campaign Manager record with episode ID despite episode record retrieval failure")
                    except Exception as update_error:
                        logger.error(f"Failed to update Campaign Manager record: {update_error}")
                        
                    return result
            except Exception as episode_error:
                # Handle permission issues or other Airtable errors
                error_message = str(episode_error)
                logger.error(f"Error retrieving episode record {selected_episode_id}: {error_message}")
                
                result['error_reason'] = f"Error retrieving episode record: {error_message}"
                result['episode_id'] = selected_episode_id  # Still include the episode ID
                result['execution_time'] = time.time() - start_time
                result['tokens_used'] = episode_tokens['total']
                    
                return result
            
            episode_fields = episode_record.get('fields', {})
            episode_title = episode_fields.get('Episode Title', '')
            episode_summary = episode_fields.get('Summary', '')
            episode_ai_summary = episode_fields.get('AI Summary', '')
            
            # Step 7: Run pitch writing
            pitch_result, pitch_tokens, pitch_time = await self._run_pitch_writing(
                client_name, podcast_name, episode_title, episode_summary, episode_ai_summary, angles, podcast_id
            )
            
            if not pitch_result:
                result['error_reason'] = "Failed to generate pitch topics"
                result['episode_id'] = selected_episode_id  # Still include the episode ID
                result['execution_time'] = time.time() - start_time
                result['tokens_used'] = episode_tokens['total'] + pitch_tokens['total']
                return result
            
            # Step 8: Format the pitch topics in the same format as the original script
            pitch_topics_formatted = f"""
1. {pitch_result.topic_1} : {pitch_result.description_1}
2. {pitch_result.topic_2} : {pitch_result.description_2}
3. {pitch_result.topic_3}: {pitch_result.description_3}
            """
            
            # Step 9: Update the Campaign Manager record
            try:
                update_fields = {
                    'Status': 'Episode and angles selected',
                    'Pitch Episode': selected_episode_id,
                    'Pitch Topics': pitch_topics_formatted
                }
                
                self.airtable_service.update_record(self.CAMPAIGN_MANAGER_TABLE, cm_record_id, update_fields)
            except Exception as update_error:
                logger.error(f"Failed to update Campaign Manager record: {update_error}")
                result['error_reason'] = f"Failed to update Campaign Manager record: {str(update_error)}"
                result['episode_id'] = selected_episode_id
                result['pitch_topics'] = pitch_topics_formatted
                result['execution_time'] = time.time() - start_time
                result['tokens_used'] = episode_tokens['total'] + pitch_tokens['total']
                result['status'] = 'PartialSuccess'  # We got the results but couldn't update Airtable
                return result
            
            # Update result with success data
            result.update({
                'status': 'Success',
                'episode_id': selected_episode_id,
                'pitch_topics': pitch_topics_formatted,
                'execution_time': time.time() - start_time,
                'tokens_used': episode_tokens['total'] + pitch_tokens['total']
            })
            
            logger.info(f"Successfully processed record {cm_record_id}, selected episode {selected_episode_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing record {cm_record_id}: {e}", exc_info=True)
            result['error_reason'] = str(e)
            result['execution_time'] = time.time() - start_time
            return result
    
    async def process_batch(self, batch_records: List[Dict], semaphore, stop_flag=None) -> List[Dict]:
        """
        Process a batch of Campaign Manager records with concurrency control.
        
        Args:
            batch_records: List of record dictionaries to process
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
                logger.info("Stop flag is set - terminating batch processing")
                break
                
            async with semaphore:
                await asyncio.sleep(request_delay)
                task = asyncio.create_task(self.process_single_record(record['id']))
                tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    async def process_all_records(self, max_concurrency=2, batch_size=4, stop_flag=None) -> Dict[str, Any]:
        """
        Process all Campaign Manager records in the 'Fit' view.
        
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
                
            # Fetch records from the 'Fit' view
            logger.info(f"Fetching records from the '{self.FIT_VIEW}' view")
            records = self.airtable_service.get_records_from_view(
                self.CAMPAIGN_MANAGER_TABLE, self.FIT_VIEW)
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
            logger.info(f"  Total tokens used: {stats['total_tokens']}")
            logger.info(f"  Average tokens per record: {stats['total_tokens']/max(stats['successful'], 1):.1f}")
            logger.info(f"  Total processing duration: {stats['duration_seconds']:.2f} seconds")
            logger.info(f"  Stopped early: {stats['stopped_early']}")
            logger.info("-----------------------------")
            
            # Save stats to file
            stats_file = f"pitch_episode_stats_{timestamp}.json"
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
async def pitch_episode_selection_async(stop_flag: Optional[Any] = None) -> Dict[str, Any]:
    """
    Async entry point for pitch_episode_selection script.
    
    Args:
        stop_flag: Optional event to signal when to stop processing
        
    Returns:
        Dictionary with processing statistics
    """
    logger.info("Starting Pitch Episode Selection Automation (Optimized)")
    
    try:
        processor = PitchEpisodeProcessor()
        
        # Check if should stop before starting
        if stop_flag and stop_flag.is_set():
            logger.info("Stop flag set before starting processing")
            return {'status': 'stopped', 'message': 'Processing stopped by stop flag', 'stopped_early': True}
        
        # Process all records (with default concurrency and batch size)
        stats = await processor.process_all_records(max_concurrency=2, batch_size=5, stop_flag=stop_flag)
        
        return stats
    except Exception as e:
        logger.critical(f"Pitch Episode Selection automation failed: {e}", exc_info=True)
        return {'status': 'error', 'message': str(e)}


# Synchronous wrapper for compatibility with existing code
def pitch_episode_selection(stop_flag: Optional[Any] = None) -> Dict[str, Any]:
    """
    Synchronous wrapper for pitch_episode_selection_async.
    
    Args:
        stop_flag: Optional event to signal when to stop processing
        
    Returns:
        Dictionary with processing statistics
    """
    return asyncio.run(pitch_episode_selection_async(stop_flag))


# Direct execution entry point
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    logger.info("=============================================")
    logger.info("Starting Pitch Episode Selection Process (Optimized)")
    logger.info("=============================================")
    
    start_run_time = time.time()
    results = asyncio.run(pitch_episode_selection_async())
    end_run_time = time.time()
    
    total_run_duration = end_run_time - start_run_time
    logger.info("=============================================")
    logger.info("Pitch Episode Selection Process Ended")
    logger.info(f"Total script execution time: {total_run_duration:.2f} seconds")
    logger.info("=============================================") 