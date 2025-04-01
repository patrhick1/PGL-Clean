"""
Pitch Writer Script (Optimized)

This module completes the final step of creating pitch emails. It fetches data 
from Airtable (e.g., client details, podcast/episode info, angles) and uses
LangChain with Anthropic Claude Haiku to generate a tailored pitch and subject line.
Finally, it updates the "Campaign Manager" table with the newly created pitch.

This optimized version uses:
- Class-based approach for better code organization
- Asynchronous processing for improved performance
- LangChain for structured outputs and more robust API interactions
- Batch processing capabilities
- Enhanced error handling and retry mechanisms
- Detailed metrics tracking

Author: Paschal Okonkwor
Date: 2025-01-06
"""

import os
import json
import time
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pydantic import BaseModel, Field

# Airtable service
from airtable_service import PodcastService

# LangChain imports
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate

# Add anthropic for tokenizer
import anthropic

# AI usage tracker
from ai_usage_tracker import tracker as ai_tracker

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class PitchEmail(BaseModel):
    """Pitch email response model"""
    email_body: str = Field(
        ..., 
        description="The complete pitch email body text, ready to be sent to the podcast host"
    )


class SubjectLine(BaseModel):
    """Subject line response model"""
    subject: str = Field(
        ..., 
        description="A clear, concise, and engaging email subject line"
    )


class PitchWriterProcessor:
    """
    Handles the process of creating pitch emails for podcast outreach.
    Fetches data from Airtable, generates personalized pitches using Claude,
    and updates records with the results.
    """

    def __init__(self):
        """Initialize services and configurations needed for pitch writing"""
        self.airtable_client = PodcastService()
        
        # Initialize LangChain with Claude Haiku
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            anthropic_api_key=os.getenv('ANTHROPIC_API'),
            temperature=0.4,
            max_tokens=2000
        )
        
        # Store the model name for consistent usage tracking
        self.model_name = "claude-3-5-sonnet-20241022"
        
        # Initialize Anthropic tokenizer for accurate token counting
        self.anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API'))
        
        # Configure constants for the process
        self.CAMPAIGN_MANAGER_TABLE_NAME = 'Campaign Manager'
        self.CAMPAIGNS_TABLE_NAME = 'Campaigns'
        self.PODCASTS_TABLE_NAME = 'Podcasts'
        self.PODCAST_EPISODES_TABLE_NAME = 'Podcast_Episodes'
        self.EPISODE_AND_ANGLES_VIEW = 'Episode and angles'
        
        logger.info("PitchWriterProcessor initialized")

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens accurately using the Anthropic tokenizer
        
        Args:
            text: The text to tokenize and count
            
        Returns:
            Number of tokens in the text
        """
        try:
            # Use the beta.messages.count_tokens method instead of direct client.count_tokens
            count_result = self.anthropic_client.beta.messages.count_tokens(
                model=self.model_name,
                messages=[
                    {
                        "role": "user", 
                        "content": text
                    }
                ]
            )
            return count_result.input_tokens
        except Exception as e:
            logger.warning(f"Error counting tokens: {str(e)}")
            # Fallback to rough approximation if tokenizer fails
            return len(text) // 4

    def _create_pitch_email_template(self) -> PromptTemplate:
        """
        Create the prompt template for generating the pitch email
        """
        template = """
        You are writing a pitch email for a podcast guest appearance. Your goal is to craft a personalized, compelling
        pitch that will make the podcast host want to interview the proposed guest.

        Here are the details:

        PODCAST INFORMATION
        - Podcast Name: {podcast_name}
        - Host Name: {host_name}
        - Episode Title (for reference): {episode_title}
        - Episode Summary: {episode_summary}
        - AI Summary: {ai_summary}

        CLIENT INFORMATION
        - Client Name: {client_name}
        - Client Bio: {client_bio}
        - Client Bio Summary: {client_bio_summary}

        PITCH TOPICS
        {pitch_topics}

        GUIDELINES:
        1. Start with a friendly, personalized greeting
        2. Mention you've listened to the podcast, referencing specific content when possible
        3. If there's a recent episode with a guest, highlight that you enjoyed it
        4. Briefly introduce your client with 1-2 key credentials
        5. Clearly state that you're pitching them as a guest
        6. Outline 2-3 specific topics they can discuss (use the provided pitch topics)
        7. Explain why your client is a good fit for this podcast
        8. Keep the email concise (250-300 words)
        9. End with a clear call to action
        10. Add a professional sign-off

        IMPORTANT: ONLY write the pitch email text, nothing else. Do not include the subject line.
        Use a conversational, authentic tone that doesn't sound templated.
        """

        # Create and return the prompt template
        return PromptTemplate(
            template=template,
            input_variables=[
                "podcast_name", "host_name", "episode_title", "episode_summary", 
                "ai_summary", "client_name", "client_bio", "client_bio_summary", 
                "pitch_topics"
            ]
        )

    def _create_subject_line_template(self) -> PromptTemplate:
        """
        Create the prompt template for generating the email subject line
        """
        template = """
        Create a compelling subject line for a podcast guest pitch email. The subject line should be:
        - Concise (6-9 words)
        - Attention-grabbing
        - Specific to the podcast topic
        - Not clickbait or overly salesy

        The email pitches a potential guest for this podcast.

        PODCAST INFORMATION:
        Episode Summary: {episode_summary}
        AI Summary: {ai_summary}
        
        Only return the subject line text, nothing else.
        """

        # Create and return the prompt template
        return PromptTemplate(
            template=template,
            input_variables=["episode_summary", "ai_summary"]
        )

    async def _run_pitch_email_generation(
        self,
        podcast_name: str,
        host_name: str,
        episode_title: str,
        episode_summary: str,
        ai_summary: str,
        client_name: str,
        client_bio: str,
        client_bio_summary: str,
        pitch_topics: str,
        podcast_id: str
    ) -> Tuple[Optional[str], Dict[str, int], float]:
        """
        Run the LLM to generate a pitch email based on provided details
        
        Returns:
            Tuple containing:
            - The generated pitch email text
            - Token usage statistics
            - Execution time in seconds
        """
        start_time = time.time()
        token_usage = {"input_tokens": 0, "output_tokens": 0}
        
        # Maximum retries for API calls
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Create the chain with the pitch email template
                prompt = self._create_pitch_email_template()
                
                # Build the chain for getting structured output
                chain = (
                    prompt
                    | self.llm.with_structured_output(PitchEmail)
                )
                
                # Prepare the inputs
                inputs = {
                    "podcast_name": podcast_name,
                    "host_name": host_name,
                    "episode_title": episode_title,
                    "episode_summary": episode_summary,
                    "ai_summary": ai_summary,
                    "client_name": client_name,
                    "client_bio": client_bio,
                    "client_bio_summary": client_bio_summary,
                    "pitch_topics": pitch_topics
                }
                
                # Format the prompt to count tokens
                formatted_prompt = prompt.format(**inputs)
                
                # Execute the chain
                result = await chain.ainvoke(inputs)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Get accurate token counts using the helper method
                prompt_tokens = self._count_tokens(formatted_prompt)
                completion_tokens = self._count_tokens(result.email_body)
                
                token_usage = {"input_tokens": prompt_tokens, "output_tokens": completion_tokens}
                
                # Log usage
                ai_tracker.log_usage(
                    workflow="pitch_writer",
                    model=self.model_name,
                    tokens_in=token_usage["input_tokens"],
                    tokens_out=token_usage["output_tokens"],
                    execution_time=execution_time,
                    endpoint="langchain.anthropic",
                    podcast_id=podcast_id
                )
                
                return result.email_body, token_usage, execution_time
                
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{max_retries} failed for pitch email generation: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"All attempts failed for pitch email generation: {str(e)}")
                    return None, token_usage, time.time() - start_time
    
    async def _run_subject_line_generation(
        self,
        episode_summary: str,
        ai_summary: str,
        podcast_id: str
    ) -> Tuple[Optional[str], Dict[str, int], float]:
        """
        Run the LLM to generate an email subject line
        
        Returns:
            Tuple containing:
            - The generated subject line
            - Token usage statistics
            - Execution time in seconds
        """
        start_time = time.time()
        token_usage = {"input_tokens": 0, "output_tokens": 0}
        
        # Maximum retries for API calls
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Create the chain with the subject line template
                prompt = self._create_subject_line_template()
                
                # Build the chain for getting structured output
                chain = (
                    prompt
                    | self.llm.with_structured_output(SubjectLine)
                )
                
                # Prepare the inputs
                inputs = {
                    "episode_summary": episode_summary,
                    "ai_summary": ai_summary
                }
                
                # Format the prompt to count tokens
                formatted_prompt = prompt.format(**inputs)
                
                # Execute the chain
                result = await chain.ainvoke(inputs)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Get accurate token counts using the helper method
                prompt_tokens = self._count_tokens(formatted_prompt)
                completion_tokens = self._count_tokens(result.subject)
                
                token_usage = {"input_tokens": prompt_tokens, "output_tokens": completion_tokens}
                
                # Log usage
                ai_tracker.log_usage(
                    workflow="pitch_writer_subject",
                    model=self.model_name,
                    tokens_in=token_usage["input_tokens"],
                    tokens_out=token_usage["output_tokens"],
                    execution_time=execution_time,
                    endpoint="langchain.anthropic",
                    podcast_id=podcast_id
                )
                
                return result.subject, token_usage, execution_time
                
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{max_retries} failed for subject line generation: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"All attempts failed for subject line generation: {str(e)}")
                    return None, token_usage, time.time() - start_time

    async def process_single_record(self, cm_record_id: str) -> Dict[str, Any]:
        """
        Process a single Campaign Manager record to generate a pitch email.
        
        Args:
            cm_record_id: The ID of the Campaign Manager record to process
            
        Returns:
            Dictionary with processing results and statistics
        """
        result = {
            "record_id": cm_record_id,
            "success": False,
            "source_used": "none",
            "status": "error",
            "execution_time": 0,
            "error_reason": "",
            "token_usage": {"input_tokens": 0, "output_tokens": 0}
        }
        
        start_time = time.time()
        
        try:
            # Get the Campaign Manager record
            cm_record = self.airtable_client.get_record(self.CAMPAIGN_MANAGER_TABLE_NAME, cm_record_id)
            if not cm_record:
                result["error_reason"] = f"Campaign Manager record {cm_record_id} not found"
                return result
                
            cm_fields = cm_record.get('fields', {})
            
            # Get campaign information
            campaign_ids = cm_fields.get('Campaigns', [])
            if not campaign_ids:
                result["error_reason"] = f"No campaign linked to Campaign Manager record {cm_record_id}"
                return result
                
            campaign_id = campaign_ids[0]
            campaign_record = self.airtable_client.get_record(self.CAMPAIGNS_TABLE_NAME, campaign_id)
            campaign_fields = campaign_record.get('fields', {})
            
            # Extract campaign data
            bio = campaign_fields.get('TextBio', '')
            bio_summary = campaign_fields.get('SummaryBio', '')
            client_names = campaign_fields.get('Name (from Client)', [])
            client_name = client_names[0] if client_names else 'No Client Name'
            
            # Get podcast information
            podcast_ids = cm_fields.get('Podcast Name', [])
            if not podcast_ids:
                result["error_reason"] = f"No podcast linked to Campaign Manager record {cm_record_id}"
                return result
                
            podcast_id = podcast_ids[0]
            podcast_record = self.airtable_client.get_record(self.PODCASTS_TABLE_NAME, podcast_id)
            podcast_fields = podcast_record.get('fields', {})
            podcast_name = podcast_fields.get('Podcast Name', '')
            host_name = podcast_fields.get('Host Name', '')
            
            # Get episode information
            podcast_episode_id = cm_fields.get('Pitch Episode', '')
            if not podcast_episode_id:
                result["error_reason"] = f"No podcast episode linked to Campaign Manager record {cm_record_id}"
                return result
                
            podcast_episode_record = self.airtable_client.get_record(self.PODCAST_EPISODES_TABLE_NAME, podcast_episode_id)
            podcast_episode_field = podcast_episode_record.get('fields', {})
            guest_name = podcast_episode_field.get('Guest', '')
            episode_title = podcast_episode_field.get('Episode Title', '')
            episode_summary = podcast_episode_field.get('Summary', '')
            episode_ai_summary = podcast_episode_field.get('AI Summary', '')
            
            # Get the pitch topics
            pitch_topics = cm_fields.get('Pitch Topics', '')
            
            # Generate the pitch email
            pitch_email, email_tokens, email_exec_time = await self._run_pitch_email_generation(
                podcast_name=podcast_name,
                host_name=host_name,
                episode_title=episode_title,
                episode_summary=episode_summary,
                ai_summary=episode_ai_summary,
                client_name=client_name,
                client_bio=bio,
                client_bio_summary=bio_summary,
                pitch_topics=pitch_topics,
                podcast_id=podcast_id
            )
            
            if not pitch_email:
                result["error_reason"] = "Failed to generate pitch email"
                return result
            
            # Generate or set the subject line
            if guest_name:
                # If there's a guest, use a simple template
                subject_line = f"Great episode with {guest_name}"
                subject_tokens = {"input_tokens": 0, "output_tokens": 0}
                subject_exec_time = 0
            else:
                # Otherwise, generate a custom subject line
                subject_line, subject_tokens, subject_exec_time = await self._run_subject_line_generation(
                    episode_summary=episode_summary,
                    ai_summary=episode_ai_summary,
                    podcast_id=podcast_id
                )
                
                if not subject_line:
                    result["error_reason"] = "Failed to generate subject line"
                    return result
            
            # Update the Airtable record
            update_fields = {
                'Status': 'Pitch Done',
                'Pitch Email': pitch_email,
                'Subject Line': subject_line
            }
            
            update_result = self.airtable_client.update_record(
                self.CAMPAIGN_MANAGER_TABLE_NAME,
                cm_record_id,
                update_fields
            )
            
            if not update_result:
                result["error_reason"] = "Failed to update Airtable record"
                return result
            
            # Calculate combined token usage
            total_tokens = {
                "input_tokens": email_tokens["input_tokens"] + subject_tokens["input_tokens"],
                "output_tokens": email_tokens["output_tokens"] + subject_tokens["output_tokens"]
            }
            
            # Set success result
            result["success"] = True
            result["status"] = "Pitch Done"
            result["source_used"] = "both" if guest_name else "episode"
            result["token_usage"] = total_tokens
            result["execution_time"] = time.time() - start_time
            
            logger.info(f"Successfully processed Campaign Manager record {cm_record_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing Campaign Manager record {cm_record_id}: {str(e)}")
            result["error_reason"] = str(e)
            result["execution_time"] = time.time() - start_time
            return result

    async def process_batch(self, batch_records: List[Dict], semaphore) -> List[Dict]:
        """
        Process a batch of Campaign Manager records with concurrency control
        
        Args:
            batch_records: List of Campaign Manager records to process
            semaphore: AsyncIO semaphore for concurrency control
            
        Returns:
            List of processing results
        """
        tasks = []
        for record in batch_records:
            record_id = record['id']
            # Process each record with the semaphore to limit concurrency
            tasks.append(self._process_with_semaphore(record_id, semaphore))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        return results
    
    async def _process_with_semaphore(self, record_id: str, semaphore) -> Dict[str, Any]:
        """Helper method to process a record with semaphore control"""
        async with semaphore:
            return await self.process_single_record(record_id)

    async def process_all_records(self, max_concurrency=1, batch_size=1) -> Dict[str, Any]:
        """
        Process all Campaign Manager records from the "Episode and angles" view
        
        Args:
            max_concurrency: Maximum number of concurrent API calls
            batch_size: Number of records to process in each batch
            
        Returns:
            Dictionary with processing statistics
        """
        start_time = time.time()
        
        # Overall statistics to track
        stats = {
            "total_records": 0,
            "successful": 0,
            "failed": 0,
            "total_tokens": {"input_tokens": 0, "output_tokens": 0},
            "total_execution_time": 0,
            "records_processed": []
        }

        try:
            logger.info(f"Starting pitch writer process with concurrency={max_concurrency}, batch_size={batch_size}")
            
            # Fetch all records from the Episode and angles view
            campaign_manager_records = self.airtable_client.get_records_from_view(
                self.CAMPAIGN_MANAGER_TABLE_NAME, 
                self.EPISODE_AND_ANGLES_VIEW
            )
            
            if not campaign_manager_records:
                logger.info("No records found in 'Episode and angles' view")
                return {
                    **stats,
                    "message": "No records found to process",
                    "timestamp": datetime.now().isoformat()
                }
            
            stats["total_records"] = len(campaign_manager_records)
            logger.info(f"Found {stats['total_records']} records to process")
            
            # Create batches of records
            batches = [
                campaign_manager_records[i:i + batch_size] 
                for i in range(0, len(campaign_manager_records), batch_size)
            ]
            
            logger.info(f"Split into {len(batches)} batches")
            
            # Create a semaphore to limit concurrent API calls
            semaphore = asyncio.Semaphore(max_concurrency)
            
            # Process each batch
            all_results = []
            for i, batch in enumerate(batches):
                logger.info(f"Processing batch {i+1}/{len(batches)}")
                batch_results = await self.process_batch(batch, semaphore)
                all_results.extend(batch_results)
            
            # Calculate final statistics
            for result in all_results:
                if result["success"]:
                    stats["successful"] += 1
                else:
                    stats["failed"] += 1
                
                stats["total_tokens"]["input_tokens"] += result["token_usage"]["input_tokens"]
                stats["total_tokens"]["output_tokens"] += result["token_usage"]["output_tokens"]
                stats["records_processed"].append(result)
            
            stats["total_execution_time"] = time.time() - start_time
            
            # Save statistics to file for reference
            stats_file = f"pitch_writer_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(stats_file, "w") as f:
                json.dump({
                    **stats,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
            
            logger.info(f"Pitch writer completed. Success: {stats['successful']}/{stats['total_records']}. Stats saved to {stats_file}")
            
            return {
                **stats,
                "message": "Processing completed successfully",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in process_all_records: {str(e)}")
            stats["total_execution_time"] = time.time() - start_time
            return {
                **stats,
                "error": str(e),
                "message": "Processing failed with an error",
                "timestamp": datetime.now().isoformat()
            }

async def pitch_writer_async(stop_flag: Optional[Any] = None) -> Dict[str, Any]:
    """
    Async entry point for pitch writer processing
    
    Args:
        stop_flag: Optional threading.Event that signals when to stop processing
        
    Returns:
        Dictionary with processing statistics
    """
    try:
        processor = PitchWriterProcessor()
        
        # Process with modest concurrency/batch settings
        return await processor.process_all_records(max_concurrency=2, batch_size=2)
        
    except Exception as e:
        logger.error(f"Error in pitch_writer_async: {str(e)}")
        return {
            "error": str(e),
            "message": "Processing failed with an error",
            "timestamp": datetime.now().isoformat()
        }

def pitch_writer(stop_flag: Optional[Any] = None) -> Dict[str, Any]:
    """
    Synchronous entry point for the pitch writer process
    
    Args:
        stop_flag: Optional threading.Event that signals when to stop processing
        
    Returns:
        Dictionary with processing statistics
    """
    try:
        # Create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async function in the new loop
            result = loop.run_until_complete(pitch_writer_async(stop_flag))
            return result
        finally:
            # Clean up the loop
            loop.close()
    
    except Exception as e:
        logger.error(f"Error in pitch_writer: {str(e)}")
        return {
            "error": str(e),
            "message": "Processing failed with an error",
            "timestamp": datetime.now().isoformat()
        }

# For command-line execution
if __name__ == "__main__":
    result = pitch_writer()
    print(f"Processed {result['total_records']} records.")
    print(f"Success: {result['successful']}, Failed: {result['failed']}") 