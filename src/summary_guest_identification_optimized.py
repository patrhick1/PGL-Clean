#!/usr/bin/env python
"""
Summary Guest Identification Optimized

This script processes podcast episodes from Airtable to identify hosts and guests.
It analyzes podcast summaries first, falling back to transcriptions if needed.
Results are updated directly in the Airtable database.

Key features:
- Prioritizes summary analysis, with fallback to transcription
- Controls concurrency to respect API rate limits
- Implements robust retry logic with exponential backoff
- Saves comprehensive statistics and logs detailed progress

Usage:
- Run directly: python summary_guest_identification_optimized.py
- Or import the process_summary_host_guest function for integration with other scripts
"""
import os
import json
import logging
import re
import time
from dotenv import load_dotenv
import asyncio
from typing import Optional, Union, List, Dict, Any, Tuple
import pandas as pd
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ValidationError
import threading
import uuid
import random


# Import LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain.prompts import PromptTemplate

# Assuming these modules exist and are correctly set up
from .airtable_service import PodcastService
from .ai_usage_tracker import tracker as ai_tracker


# Add tiktoken for OpenAI token counting
try:
    import tiktoken
except ImportError:
    tiktoken = None

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Pydantic models for response validation
class ValidationInfo(BaseModel):
    confidence: float = Field(description="Your confidence level in the host/guest identification, from 0 to 1")
    evidence: str = Field(..., description="Quotes or specific context from the summary that supports your identification")

class TopicPoint(BaseModel):
    name: str = Field(..., description="Name of a main topic discussed in the podcast")
    key_points: List[str] = Field(default_factory=list, description="List of key points discussed within this topic")

class PodcastAnalysis(BaseModel):
    host: Union[str, List[str], None] = Field(None, description="The actual name(s) of the podcast episode host(s). If you cannot determine the actual name, leave this as null.")
    guest: Union[str, List[str], None] = Field(None, description="The actual name(s) of the podcast episode guest(s). If you cannot determine the actual name, leave this as null.")
    status: str = Field(..., enum=["Both", "Host", "Guest", "None"], description="Whether both host and guest were identified (Both), only host (Host), only guest (Guest), or neither (None)")
    summary: str = Field(..., description="A detailed understanding of the podcast episode") 
    topics: List[TopicPoint] = Field(..., description="List of main topics discussed in the podcast")
    validation: ValidationInfo = Field(..., description="Information about the confidence level and evidence for host/guest identification")

    @field_validator('status')
    def validate_status(cls, v):
        if v not in {'Both', 'Host', 'Guest', 'None'}:
            raise ValueError(f'Status must be one of: Both, Host, Guest, None')
        return v

class PodcastProcessor:
    def __init__(self):
        try:
            self.airtable = PodcastService()
            self.parser = PydanticOutputParser(pydantic_object=PodcastAnalysis)
            
            # Ensure API key is loaded with better error handling
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set. Please set this in your environment or .env file.")
            
            # === MODEL CONFIGURATION ===
            # This script is configured to use only Gemini 2.0 Flash
            # If you need to switch models, modify this section
            self.model_name = "gemini-2.0-flash"
                
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name, 
                google_api_key=api_key,
                temperature=0.4,
                request_options={"timeout": 180} # Explicit timeout for the request
            )
            self.prompt_template = self._create_prompt_template()
        except Exception as e:
            logger.critical(f"Failed to initialize PodcastProcessor: {e}", exc_info=True)
            raise

    def _create_prompt_template(self):
        """Create the prompt template for podcast analysis."""
        try:
            # Get format instructions with error handling
            format_instructions = self.parser.get_format_instructions()
            
            return PromptTemplate(
                 template="""You are a podcast analysis expert. Analyze the following podcast episode text (which could be a summary or a transcription) to extract:

1.  The ACTUAL NAMES of hosts (no generic terms like 'Host').
2.  The ACTUAL NAMES of the guests (no generic terms like 'Guest').
3.  A detailed understanding of the content (generate a concise summary based *only* on the provided text).
4.  The main topics discussed.

Episode Title: {episode_title}

Provided Text:
{summary}

IMPORTANT INSTRUCTIONS:

*   Focus on identifying the guest based on phrases like "joining us today," "interview with," "in conversation with," "fellow," "expert," "featured speaker," or similar introductions within the 'Provided Text'. A guest is someone participating in the discussion who is not presented as a regular host.
*   For host and guest fields, extract ONLY actual names mentioned. If names are unclear or not mentioned, use null.
*   Do NOT use generic placeholders like "Host" or "Guest" - we need real names or null.
*   If someone is identified as "Pastor Jake" or similar, use the full name/title found in the text.
*   The status field indicates whether you identified: Both host and guest, only Host, only Guest, or None based on the names found.
*   The 'summary' field in your output should be YOUR generated summary of the 'Provided Text', capturing the essence of the discussion.
*   For the validation field, provide specific evidence (quotes if possible) from the 'Provided Text' that supports your host/guest identification. Include the *exact* phrase that led you to believe someone is a guest.
*   If actual names are not clearly identified, return null for those fields.

**Example 1 (Multiple Guests):**

Provided Text: "Each month, show co-host John Pasalis presents his latest data and top stories on Toronto area real estate in conversation with fellow Realosophy agents, Gus Papaioannou and Davin McMahon, and our Move Smartly mortgage expert, David Larock - and you, our viewers, who join the conversation in our live chat hosted by Move Smartly Editor Urmi Desai to make this is one of our liveliest segments!"
Host: John Pasalis, Urmi Desai
Guest: Gus Papaioannou, Davin McMahon, David Larock
Status: Both
Validation: {{"confidence": 0.95, "evidence": "The text clearly identifies John Pasalis as 'show co-host' and Urmi Desai as 'Move Smartly Editor' who hosts the live chat. Gus Papaioannou and Davin McMahon are identified as 'fellow Realosophy agents' and David Larock as 'our Move Smartly mortgage expert', indicating they are guests."}}

The output should follow this format:
{format_instructions}""",
                input_variables=["episode_title", "summary", "format_instructions"],
            )
        except Exception as e:
            logger.error(f"Failed to create prompt template: {e}", exc_info=True)
            raise

    async def _run_llm_analysis(
        self,
        llm: ChatGoogleGenerativeAI,
        prompt_template: PromptTemplate,
        parser: PydanticOutputParser,
        episode_title: str,
        text_to_analyze: str,
        record_id: str,
        source_name: str # e.g., "Summary" or "Transcription"
    ) -> Tuple[Optional[PodcastAnalysis], Dict[str, int], float]:
        """
        Run LLM analysis on the provided text with retry logic.
        
        Args:
            llm: The LLM to use for analysis
            prompt_template: The prompt template to use
            parser: The Pydantic parser for structured output
            episode_title: The title of the podcast episode
            text_to_analyze: The text content to analyze (summary or transcription)
            record_id: The Airtable record ID being processed
            source_name: The source of the text (Summary or Transcription)
            
        Returns:
            Tuple containing the PodcastAnalysis object, token usage statistics, and execution time
        """
        if not text_to_analyze or len(text_to_analyze.strip()) < 50:
            logger.warning(f"Text too short for analysis: {len(text_to_analyze.strip()) if text_to_analyze else 0} chars")
            return None, {}, 0.0
            
        start_time = time.time()
        formatted_prompt = ""
        try:
            # Get format instructions for the prompt
            format_instructions = self.parser.get_format_instructions()
            
            # Prepare prompt
            formatted_prompt = prompt_template.format(
                episode_title=episode_title,
                summary=text_to_analyze, # Prompt uses 'summary' variable name, but we pass the text
                format_instructions=format_instructions
            )
            
            # Initialize retry parameters
            max_retries = 5
            retry_count = 0
            base_delay = 5  # Start with 5 seconds
            
            # Get model name for delay settings
            model_name = getattr(llm, 'model', getattr(llm, 'model_name', '')).lower()
            
            # For Claude models, use longer delays
            if "claude" in model_name:
                base_delay = 15  # Start with 15 seconds for Claude
            
            # Try to get a valid response with retries
            while retry_count < max_retries:
                try:
                    # Structure output using the PodcastAnalysis model
                    llm_with_output = llm.with_structured_output(PodcastAnalysis)
                    
                    # Run in a separate thread to avoid blocking
                    llm_response = await asyncio.to_thread(llm_with_output.invoke, formatted_prompt)
                    
                    # Extract token usage info using improved method
                    token_usage = self._extract_token_usage(llm_response, formatted_prompt)
                    
                    # Return the analysis result and token usage
                    execution_time = time.time() - start_time
                    return llm_response, token_usage, execution_time
                
                except (ValidationError, ValueError) as parse_error:
                    # Parsing/validation errors may be worth retrying
                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.error(f"Max retries reached after parsing errors: {parse_error}")
                        raise
                    
                    wait_time = base_delay * (2 ** (retry_count - 1))
                    logger.warning(f"Parsing error on attempt {retry_count}, retrying in {wait_time}s: {parse_error}")
                    await asyncio.sleep(wait_time)
                
                except Exception as e:
                    retry_count += 1
                    error_str = str(e).lower()
                    
                    # Handle rate limits and server errors differently with more aggressive backoff
                    is_rate_limit = any(x in error_str for x in ["rate", "limit", "429", "quota"])
                    is_server_error = any(x in error_str for x in ["500", "503", "server"])
                    
                    if (is_rate_limit or is_server_error) and retry_count < max_retries:
                        # More aggressive backoff for rate limits
                        multiplier = 3 if is_rate_limit else 2
                        wait_time = base_delay * (multiplier ** retry_count)
                        
                        # Add random jitter to prevent thundering herd
                        jitter = random.uniform(0.5, 1.5)
                        wait_time = wait_time * jitter
                        
                        logger.warning(f"Rate limit/server error. Retry {retry_count}/{max_retries} in {wait_time:.1f}s")
                        await asyncio.sleep(wait_time)
                    elif retry_count >= max_retries:
                        logger.error(f"Max retries ({max_retries}) reached. Last error: {e}")
                        raise
                    else:
                        logger.error(f"Non-retryable error: {e}")
                    raise
        
        except Exception as e:
            logger.error(f"Failed to analyze content: {e}", exc_info=True)
            execution_time = time.time() - start_time
            return None, {}, execution_time

    def _extract_token_usage(self, response: Any, prompt_text: str = "") -> Dict[str, int]:
        """
        Extract token usage information from the LLM response with provider-specific methods.
        
        Args:
            response: The LLM response object
            prompt_text: The original prompt text (for estimation if needed)
            
        Returns:
            Dict containing token usage statistics (prompt_tokens, completion_tokens, total_tokens)
        """
        token_usage = {}
        
        try:
            # Determine which model provider we're using based on model name
            model_provider = 'unknown'
            if hasattr(self, 'model_name'):
                model_name = self.model_name.lower()
                if 'gemini' in model_name:
                    model_provider = 'google'
                elif 'claude' in model_name:
                    model_provider = 'anthropic'
                elif 'gpt' in model_name or 'o3-mini' in model_name:
                    model_provider = 'openai'
            
            # Method 1: Try to get token usage from raw_response.usage_metadata (Gemini's standard location)
            raw_response = getattr(response, '_raw_response', None)
            if raw_response:
                # For Google/Gemini models
                if model_provider == 'google':
                    usage_metadata = getattr(raw_response, 'usage_metadata', None)
                    if usage_metadata:
                        token_usage = {
                            'prompt_tokens': usage_metadata.get('prompt_token_count', 0),
                            'completion_tokens': usage_metadata.get('candidates_token_count', 0),
                            'total_tokens': usage_metadata.get('total_token_count', 0),
                            'method': 'usage_metadata'
                        }
                        logger.debug(f"Extracted token usage from Gemini usage_metadata: {token_usage}")
                        return token_usage
                
                # For OpenAI models (o3-mini, etc)
                elif model_provider == 'openai' and hasattr(raw_response, 'usage'):
                    usage = raw_response.usage
                    if usage:
                        token_usage = {
                            'prompt_tokens': usage.get('prompt_tokens', 0),
                            'completion_tokens': usage.get('completion_tokens', 0),
                            'total_tokens': usage.get('total_tokens', 0),
                            'method': 'openai_usage'
                        }
                        logger.debug(f"Extracted token usage from OpenAI usage: {token_usage}")
                        return token_usage
                
                # For Anthropic models (Claude)
                elif model_provider == 'anthropic' and hasattr(raw_response, 'usage'):
                    usage = raw_response.usage
                    if usage:
                        token_usage = {
                            'prompt_tokens': usage.get('input_tokens', 0),
                            'completion_tokens': usage.get('output_tokens', 0),
                            'total_tokens': usage.get('input_tokens', 0) + usage.get('output_tokens', 0),
                            'method': 'anthropic_usage'
                        }
                        logger.debug(f"Extracted token usage from Anthropic usage: {token_usage}")
                        return token_usage
            
            # Method 2: Provider-specific direct token counting
            try:
                # Get response content for output token counting
                response_text = ""
                if hasattr(response, 'content'):
                    response_text = response.content
                elif hasattr(response, 'model_dump_json'):
                    response_text = response.model_dump_json()
                
                # For Google/Gemini models, use direct API counting
                if model_provider == 'google' and prompt_text and response_text:
                    import google.generativeai as genai
                    
                    # Check if the API key is available for direct token counting
                    api_key = os.getenv("GEMINI_API_KEY")
                    if api_key:
                        # Configure the API
                        genai.configure(api_key=api_key)
                        
                        # Get the model and count tokens
                        model = genai.GenerativeModel("gemini-2.0-flash")
                        input_tokens = model.count_tokens(prompt_text).total_tokens
                        output_tokens = model.count_tokens(response_text).total_tokens
                        
                        token_usage = {
                            'prompt_tokens': input_tokens,
                            'completion_tokens': output_tokens,
                            'total_tokens': input_tokens + output_tokens,
                            'method': 'direct_api_count'
                        }
                        logger.debug(f"Counted Gemini tokens directly using API: {token_usage}")
                        return token_usage
                
                # For OpenAI models, use tiktoken
                elif model_provider == 'openai' and tiktoken and prompt_text and response_text:
                    # Determine encoding based on model
                    model_name = getattr(self, 'model_name', 'gpt-3.5-turbo')
                    try:
                        encoding = tiktoken.encoding_for_model(model_name)
                    except KeyError:
                        encoding = tiktoken.get_encoding("cl100k_base")  # Default for newer models
                    
                    # Count tokens
                    input_tokens = len(encoding.encode(prompt_text))
                    output_tokens = len(encoding.encode(response_text))
                    
                    token_usage = {
                        'prompt_tokens': input_tokens,
                        'completion_tokens': output_tokens,
                        'total_tokens': input_tokens + output_tokens,
                        'method': 'tiktoken_count'
                    }
                    logger.debug(f"Counted OpenAI tokens using tiktoken: {token_usage}")
                    return token_usage
                
                # For Claude/Anthropic models, use their token counter if available
                elif model_provider == 'anthropic' and prompt_text and response_text:
                    try:
                        import anthropic
                        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                        
                        # Count input tokens
                        input_count = client.count_tokens(prompt_text)
                        # Count output tokens
                        output_count = client.count_tokens(response_text)
                        
                        token_usage = {
                            'prompt_tokens': input_count,
                            'completion_tokens': output_count,
                            'total_tokens': input_count + output_count,
                            'method': 'anthropic_counter'
                        }
                        logger.debug(f"Counted Anthropic tokens using their counter: {token_usage}")
                        return token_usage
                    except (ImportError, Exception) as e:
                        logger.debug(f"Could not use Anthropic token counter: {e}")
            except Exception as e:
                logger.debug(f"Could not use direct token counting: {e}")
            
            # Method 3: Fallback to character-based estimation
            if not token_usage and prompt_text:
                # Character-based estimation (uses 4.83 chars per token for better accuracy based on test results)
                char_per_token = 4.83  # Adjusted based on empirical testing
                
                # Estimate input tokens
                input_tokens = int(len(prompt_text) / char_per_token)
                
                # Estimate output tokens
                output_text = ""
                if hasattr(response, 'content'):
                    output_text = response.content
                elif hasattr(response, 'model_dump_json'):
                    output_text = response.model_dump_json()
                else:
                    # For Pydantic models, try to get a reasonable representation
                    output_text = str(response)
                    
                output_tokens = int(len(output_text) / char_per_token)
                
                token_usage = {
                    'prompt_tokens': input_tokens,
                    'completion_tokens': output_tokens,
                    'total_tokens': input_tokens + output_tokens,
                    'method': 'character_estimation'
                }
                logger.debug(f"Estimated token usage from character count: {token_usage}")
                return token_usage
                
        except Exception as e:
            logger.warning(f"Error extracting token usage, using fallback estimation: {e}")
            
            # Emergency fallback
            if prompt_text:
                token_usage = {
                    'prompt_tokens': len(prompt_text) // 4,
                    'completion_tokens': 250,  # Default guess
                    'total_tokens': (len(prompt_text) // 4) + 250,
                    'method': 'emergency_fallback'
                }
            else:
                token_usage = {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0,
                    'method': 'no_data_available'
                }
        
        return token_usage

    async def process_single_record(self, record) -> Dict:
        """
        Process a single podcast record, first trying to use summary, then falling back to transcription.
        This updates the Airtable record with the results.
        
        Args:
            record: The Airtable record to process
            
        Returns:
            Dict containing processing results and status
        """
        start_time = time.time()
        try:
            # Extract basic record info - use 'id' key which matches Airtable's structure
            record_id = record.get('id')  # Changed from 'record_id' to 'id'
            fields = record.get('fields', {})
            
            # Get podcast_id from the linked Podcast field - this is the record ID from the Podcast table
            podcast_ids = fields.get('Podcast', [])
            podcast_id = podcast_ids[0] if podcast_ids else "Unknown"
            
            title = fields.get('Episode Title', 'Unknown Title')
            
            # Validate record ID
            if not record_id:
                logger.warning(f"Record has null ID. Cannot process. Podcast ID: {podcast_id}, Title: {title}")
                return {
                    'podcast_id': podcast_id,
                    'record_id': None,
                    'title': title,
                    'success': False,
                    'error_reason': "Record has null ID. Cannot process.",
                    'tokens': {
                        'prompt_tokens': 0,
                        'completion_tokens': 0,
                        'total_tokens': 0,
                        'method': 'none'  # Track which method was used for token counting
                    },
                    'llm_execution_time': 0.0,  # Track LLM execution time
                    'analysis_source': None  # Track which source was used
                }
                
            # Initialize result structure
            result = {
                'podcast_id': podcast_id,
                'record_id': record_id,
                'title': title,
                'success': False,
                'error_reason': None,
                'tokens': {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0,
                    'method': 'none'  # Track which method was used for token counting
                },
                'llm_execution_time': 0.0,  # Track LLM execution time
                'analysis_source': None  # Track which source was used
            }
            
            # Get summary text first - using correct field names
            summary = fields.get('Summary', '')
            transcription = fields.get('Transcription', '')
            
            analysis_source = None
            llm_execution_time = 0.0
            text_to_analyze = None # Initialize variable to hold the text we will send to the LLM
            
            # --- Determine text source --- 
            if summary and len(summary.strip()) > 1000:
                # Use Summary + Partial Transcription if both are available
                if transcription:
                    text_to_analyze = f"{summary}\n\n--- Transcription Start ---\n{transcription[:2000]}"
                    analysis_source = "Summary + Transcription (Partial)"
                    logger.info(f"Using Summary + first 2000 chars of Transcription for: {podcast_id}, record: {record_id}")
                else:
                    text_to_analyze = summary
                    analysis_source = "Summary"
                    logger.info(f"Using Summary only (Transcription missing) for: {podcast_id}, record: {record_id}")
            elif transcription and len(transcription.strip()) > 1000:
                # Fallback to Full Transcription if summary is inadequate
                text_to_analyze = transcription
                analysis_source = "Transcription (Full)"
                logger.info(f"Summary inadequate, using Full Transcription for: {podcast_id}, record: {record_id}")
            else:
                # No suitable text found
                logger.warning(f"Neither Summary nor Transcription is suitable for analysis for: {podcast_id}, record: {record_id}")
                result['error_reason'] = "No suitable content for analysis (summary/transcription too short or missing)"
                return result
            
            # --- Run Analysis --- 
            try:
                # Only run analysis if we have text_to_analyze
                if text_to_analyze:
                    analysis_result, tokens, execution_time = await self._run_llm_analysis(
                        self.llm, 
                        self.prompt_template,
                        self.parser,
                        title,
                        text_to_analyze, # Use the combined/selected text
                        record_id,
                        analysis_source # Pass the determined source name
                    )
                    
                    # Update token usage and time
                    if tokens:
                        result['tokens'] = tokens # Overwrite with the actual tokens used
                    llm_execution_time = execution_time
                    result['llm_execution_time'] = llm_execution_time
                    result['analysis_source'] = analysis_source
                else:
                    # This case should be caught by the checks above, but as a safeguard:
                    result['error_reason'] = "Internal logic error: No text was selected for analysis."
                    return result
                    
            except Exception as e:
                logger.error(f"Error during LLM analysis ({analysis_source}) for podcast {podcast_id}: {e}", exc_info=True)
                result['error_reason'] = f"LLM analysis failed using {analysis_source}: {str(e)}"
                return result

            # --- Process Analysis Result --- 
            if analysis_result is None:
                result['error_reason'] = f"LLM analysis using {analysis_source} returned no result."
                # Note: Don't return here yet, we still want to log usage below if possible
            
            # Log AI usage (attempt even if analysis_result is None, as tokens might still be recorded)
            try:
                total_execution_time = time.time() - start_time
                
                ai_tracker.log_usage(
                    workflow="summary_guest_identification",
                    model=self.model_name,
                    tokens_in=result['tokens']['prompt_tokens'],
                    tokens_out=result['tokens']['completion_tokens'],
                    execution_time=llm_execution_time,  # Use the LLM execution time for cost calculation
                    endpoint="langchain.gemini",
                    podcast_id=podcast_id
                )
                token_method = result['tokens'].get('method', 'unknown')
                logger.info(f"Logged AI usage for podcast {podcast_id}. LLM execution time: {llm_execution_time:.2f}s, "
                           f"Total processing time: {total_execution_time:.2f}s, "
                           f"Token counting method: {token_method}")
            except Exception as e:
                logger.warning(f"Failed to log AI usage for podcast {podcast_id}: {e}")
            
            # Extract the analysis result data
            if analysis_result:
                # Create the data structure for Airtable update
                update_data = {
                    'host': analysis_result.host if hasattr(analysis_result, 'host') else [],
                    'guest': analysis_result.guest if hasattr(analysis_result, 'guest') else [],
                    'validation_info': analysis_result.validation if hasattr(analysis_result, 'validation') else {'confidence': 0},
                    'tokens': result['tokens'],
                    'analysis_source': analysis_source,
                    'summary': analysis_result.summary if hasattr(analysis_result, 'summary') else None
                }
                
                # Update Airtable with the results
                update_result = await self._update_airtable_record(record_id, update_data)
                
                if update_result and update_result.get('success'):
                    result['success'] = True
                else:
                    result['error_reason'] = f"Airtable update failed: {update_result.get('error', 'Unknown error')}"
            else:
                result['error_reason'] = "Analysis completed but returned no results"
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing record: {e}", exc_info=True)
            return {
                'podcast_id': record.get('fields', {}).get('Episode ID', 'Unknown'),
                'record_id': record.get('id', 'Unknown'),
                'title': record.get('fields', {}).get('Episode Title', 'Unknown Title'),
                'success': False,
                'error_reason': str(e),
                'tokens': {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0,
                    'method': 'none'  # Track which method was used for token counting
                },
                'llm_execution_time': 0.0,  # Track LLM execution time
                'analysis_source': None  # Track which source was used
            }

    async def process_batch(self, batch, semaphore, stop_flag=None):
        """Process a batch of records concurrently with semaphore control."""
        tasks = []
        
        for record in batch:
            if stop_flag and stop_flag.is_set():
                logger.info("Stop flag set during batch processing")
                break
            # Create a task that acquires the semaphore before processing
            tasks.append(
                self._process_with_semaphore(record, semaphore, stop_flag)
            )
            
        # Wait for all processing tasks to complete and return results
        return await asyncio.gather(*tasks)
        
    async def _process_with_semaphore(self, record, semaphore, stop_flag=None):
        """Process a record with semaphore control for concurrency limitation."""
        async with semaphore:
            try:
                # Process the record and handle potential errors
                result = await self.process_single_record(record)
                return result
            except Exception as e:
                # Ensure we always return something even if processing fails
                logger.error(f"Error processing record {record.get('id', 'unknown')}: {e}", exc_info=True)
                
                # Get podcast_id from the linked Podcast field
                fields = record.get('fields', {})
                podcast_ids = fields.get('Podcast', [])
                podcast_id = podcast_ids[0] if podcast_ids else "Unknown"
                
                return {
                    'podcast_id': podcast_id,
                    'record_id': record.get('id', 'Unknown'),
                    'title': fields.get('Episode Title', 'Unknown'),
                    'success': False,
                    'error_reason': f"Unhandled error: {str(e)}",
                    'tokens': {
                        'prompt_tokens': 0,
                        'completion_tokens': 0,
                        'total_tokens': 0,
                        'method': 'none'  # Track which method was used for token counting
                    },
                    'llm_execution_time': 0.0,  # Track LLM execution time
                    'analysis_source': None  # Track which source was used
                }

    async def _update_airtable_record(self, record_id: str, analysis_results: Dict) -> Dict:
        """
        Update an Airtable record with the results of the analysis.
        
        Args:
            record_id: The Airtable record ID to update
            analysis_results: Results from the analysis, including status, confidence, hosts, guests
            
        Returns:
            Dict containing success status and error message if any
        """
        try:
            # Skip update if record_id is None or empty
            if not record_id:
                logger.warning("Cannot update Airtable record: record_id is None or empty")
                return {"success": False, "error": "Record ID is None or empty"}
                
            # Only update if we have results and confidence above threshold
            if not analysis_results or 'validation_info' not in analysis_results:
                logger.warning(f"No valid analysis results to update for record {record_id}")
                return {"success": False, "error": "No valid analysis results"}
            
            validation_info = analysis_results.get('validation_info', {})
            
            # Access confidence differently based on whether it's a Pydantic model or dict
            if hasattr(validation_info, 'confidence'):
                # It's a Pydantic model, access as attribute
                confidence = validation_info.confidence
            else:
                # It's a dictionary, access as key
                confidence = validation_info.get('confidence', 0)
            
            # Prepare fields to update - only using fields that exist in the Airtable schema
            fields = {}
            
            # Only update if confidence is high enough
            
            # Add host if present
            if 'host' in analysis_results and analysis_results['host']:
                hosts = analysis_results['host']
                if isinstance(hosts, list):
                    # Join list to string since Host is a single line text field
                    fields['Host'] = ", ".join(hosts)
                elif isinstance(hosts, str):
                    fields['Host'] = hosts
                else:
                    # Handle unexpected format
                    logger.warning(f"Host not in expected format: {hosts}")
            
            # Add guest if present
            if 'guest' in analysis_results and analysis_results['guest']:
                guests = analysis_results['guest']
                if isinstance(guests, list):
                    # Join list to string since Guest is a single line text field
                    fields['Guest'] = ", ".join(guests)
                elif isinstance(guests, str):
                    fields['Guest'] = guests
                else:
                    # Handle unexpected format
                    logger.warning(f"Guest not in expected format: {guests}")
            
            # Add flag for Guest Confirmed if we identified guests with high confidence
            if 'guest' in analysis_results and analysis_results['guest']:
                fields['Guest Confirmed'] = True
            
            # Also mark Summary Checked as true since we've analyzed it
            fields['Summary Checked'] = True
            
            # Mark the Solo flag as False if a guest is identified, True if only a host is identified
            if 'guest' in analysis_results and analysis_results['guest']:
                fields['Solo'] = False
            elif 'host' in analysis_results and analysis_results['host'] and not analysis_results.get('guest'):
                fields['Solo'] = True

            if confidence >= 0.7:
                fields['Flagged Human'] = False
            else:
                # Low confidence - mark for human review
                fields['Flagged Human'] = True  # Flag for human review
        
            # Generate an AI summary if we have a good result and it's available in the analysis results
            if 'summary' in analysis_results and analysis_results.get('summary'):
                fields['AI Summary'] = analysis_results['summary']
            
            # Perform the update
            logger.info(f"Updating Airtable record {record_id}")
            response = self.airtable.update_record("Podcast_Episodes", record_id, fields)
            
            if response and 'id' in response:
                logger.info(f"Successfully updated record {record_id}")
                return {"success": True}
            else:
                logger.error(f"Failed to update record {record_id}: {response}")
                return {"success": False, "error": "Update failed, unexpected response"}
                
        except Exception as e:
            logger.error(f"Error updating Airtable record {record_id}: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def process_all_records(self, max_concurrency=3, batch_size=5, stop_flag=None):
        """Process all podcast records asynchronously and update Airtable."""
        try:
            # Create statistics structure for tracking progress
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stats = {
                'start_time': time.time(),
                'end_time': None,
                'duration_seconds': 0,
                'total_processed': 0,
                'total_attempted': 0,
                'successful_analysis': 0,
                'failed_analysis': 0,
                'flagged': 0,
                'used_summary': 0,
                'used_transcription': 0,
                'total_prompt_tokens': 0,
                'total_completion_tokens': 0,
                'total_summary_tokens': 0,
                'total_transcription_tokens': 0,
                'grand_total_tokens': 0,
                'avg_tokens_per_record': 0,
                'total_llm_execution_time': 0.0,
                'avg_llm_execution_time': 0.0,
                'token_counting_methods': {
                    'direct_api_count': 0,
                    'usage_metadata': 0,
                    'character_estimation': 0,
                    'emergency_fallback': 0,
                    'no_data_available': 0,
                    'unknown': 0
                }
            }
            
            # Initialize results list to store all processed records
            results = []
            
            # Check stop flag immediately
            if stop_flag and stop_flag.is_set():
                logger.info("Stop flag is set - terminating process_all_records before starting")
                return {"message": "Process terminated by stop flag", "total_processed": 0, "successful": 0, "flagged": 0}

            # Adjust concurrency for Anthropic models due to their rate limits
            if hasattr(self.llm, 'model_name') and 'claude' in self.llm.model_name.lower():
                original_concurrency = max_concurrency
                max_concurrency = 1  # Force to 1 for Claude models
                logger.warning(f"Reducing concurrency from {original_concurrency} to {max_concurrency} for {self.llm.model_name} to avoid rate limits")
                # Also reduce batch size if it's too high
                if batch_size > 2:
                    original_batch_size = batch_size
                    batch_size = 2
                    logger.warning(f"Reducing batch size from {original_batch_size} to {batch_size} for {self.llm.model_name}")

            # Get records from Airtable
            table_name = "Podcast_Episodes"
            # Use the same view as the test script
            view = "No Summary"
            
            logger.info(f"Fetching records from Airtable table '{table_name}', view '{view}'...")
            records = self.airtable.get_records_from_view(table_name, view)
            logger.info(f"Found {len(records)} record(s) to potentially process.")

            # Check stop flag after fetching records
            if stop_flag and stop_flag.is_set():
                logger.info("Stop flag is set - terminating after fetching records")
                return {"message": "Process terminated by stop flag", "total_processed": 0, "successful": 0, "flagged": 0}

            if not records: 
                logger.info("No records match the criteria for processing.")
                return None

            prepared_records = []
            skipped_preparation = 0
            for record in records:  
                # Check stop flag during preparation
                if stop_flag and stop_flag.is_set():
                    logger.info("Stop flag is set - terminating during record preparation")
                    return {"message": "Process terminated by stop flag", "total_processed": 0, "successful": 0, "flagged": 0}
                
                # First ensure record has an ID - all Airtable records should have an ID
                if not record.get('id'):
                    logger.warning(f"Skipping record: Missing record ID which is required for processing.")
                    skipped_preparation += 1
                    continue
                    
                record_id = record['id']
                try:
                    fields = record['fields']
                    podcast_ids = fields.get('Podcast', [])
                    podcast_id = podcast_ids[0] if podcast_ids else None

                    # Skip if no podcast ID is found
                    if not podcast_id:
                        logger.warning(f"Skipping record {record_id}: No linked 'Podcast'.")
                        skipped_preparation += 1
                        continue
                    
                    # For optimization, check if at least one of Summary or Transcription is available
                    if not fields.get('Summary') and not fields.get('Transcription'):
                        logger.warning(f"Skipping record {record_id}: Both 'Summary' and 'Transcription' fields are empty.")
                        skipped_preparation += 1
                        continue

                    # Verification step to ensure record_id is valid
                    if not isinstance(record_id, str) or not record_id.strip():
                        logger.warning(f"Skipping record: Record ID '{record_id}' is not a valid string.")
                        skipped_preparation += 1
                        continue

                    # Add valid record to processing queue
                    prepared_records.append({
                        'id': record_id,  # Make sure we're using 'id' as the key, matching Airtable's structure
                        'fields': fields,
                        'podcast_id': podcast_id  # Store the linked Podcast record ID
                    })
                    logger.debug(f"Prepared record {record_id} for processing")
                except Exception as e:
                     logger.error(f"Error preparing record {record_id}: {e}", exc_info=True)
                     skipped_preparation += 1

            stats['total_processed'] = skipped_preparation # Start count with skipped ones

            # Check stop flag after preparation
            if stop_flag and stop_flag.is_set():
                logger.info("Stop flag is set - terminating after record preparation")
                return {"message": "Process terminated by stop flag", "total_processed": skipped_preparation, "successful": 0, "flagged": 0}

            if not prepared_records:
                 logger.warning("No records were successfully prepared for processing.")
                 # Update stats before returning
                 stats['end_time'] = time.time()
                 stats['duration_seconds'] = stats['end_time'] - stats['start_time']
                 return None

            # Process records in batches with concurrency control
            batches = [prepared_records[i:i + batch_size] for i in range(0, len(prepared_records), batch_size)]
            logger.info(f"Processing {len(prepared_records)} prepared records in {len(batches)} batches...")
            
            # Create a semaphore for concurrency control
            semaphore = asyncio.Semaphore(max_concurrency)

            for i, batch in enumerate(batches):
                # Check stop flag before each batch
                if stop_flag and stop_flag.is_set():
                    logger.info(f"Stop flag is set - terminating before processing batch {i+1}/{len(batches)}")
                    break
                
                batch_num = i + 1
                logger.info(f"--- Starting Batch {batch_num}/{len(batches)} ({len(batch)} records) ---")
                
                # Add pause between batches
                if i > 0:
                    # Pause longer after every 2 batches like in the test script
                    if batch_num % 2 == 0:
                        pause_time = 45  # 45 seconds every 2 batches
                        logger.warning(f"PAUSING for {pause_time}s after batch {batch_num}...")
                    else:
                        pause_time = 2  # 2 seconds otherwise
                        logger.info(f"Pausing for {pause_time}s before batch {batch_num}...")
                    
                    # Check stop flag during pauses
                    for _ in range(pause_time):
                        if stop_flag and stop_flag.is_set():
                            logger.info(f"Stop flag is set - terminating during pause before batch {batch_num}")
                            break
                        await asyncio.sleep(1)
                    
                    # Check again after the for loop
                    if stop_flag and stop_flag.is_set():
                        logger.info(f"Stop flag is set - terminating after pause before batch {batch_num}")
                        break

                start_batch_time = time.time()
                batch_results = await self.process_batch(batch, semaphore, stop_flag)
                batch_duration = time.time() - start_batch_time
                logger.info(f"--- Finished Batch {batch_num}/{len(batches)}. Duration: {batch_duration:.2f}s ---")

                # Process results and update stats
                for result in batch_results:
                    stats['total_processed'] += 1
                    
                    # Add result to our list
                    results.append(result)
                    
                    # Update stats based on result
                    if result.get('success'):
                        stats['successful_analysis'] += 1
                    else:
                        stats['failed_analysis'] += 1
                        logger.warning(f"Record {result.get('record_id')} analysis failed. Reason: {result.get('error_reason')}")

                    # Track token usage
                    token_data = result.get('tokens', {})
                    prompt_tokens = token_data.get('prompt_tokens', 0)
                    completion_tokens = token_data.get('completion_tokens', 0)
                    total_tokens = token_data.get('total_tokens', 0)
                    
                    stats['total_prompt_tokens'] += prompt_tokens
                    stats['total_completion_tokens'] += completion_tokens
                    stats['grand_total_tokens'] += total_tokens
                    
                    # Track token counting method
                    token_method = token_data.get('method', 'unknown')
                    if token_method in stats['token_counting_methods']:
                        stats['token_counting_methods'][token_method] += 1
                    else:
                        stats['token_counting_methods']['unknown'] += 1
                    
                    # Track LLM execution time if it's available in the result
                    if 'llm_execution_time' in result:
                        stats['total_llm_execution_time'] += result.get('llm_execution_time', 0.0)
                    
                    # Update flagged count
                    if not result.get('success'):
                        stats['flagged'] += 1
                        
                    # Track which source was used (summary or transcription)
                    analysis_source = result.get('analysis_source')
                    if analysis_source == 'Summary':
                        stats['used_summary'] += 1
                        stats['total_summary_tokens'] += total_tokens
                    elif analysis_source == 'Transcription':
                        stats['used_transcription'] += 1
                        stats['total_transcription_tokens'] += total_tokens

            # --- Post-processing & Stats ---
            stats['end_time'] = time.time()
            stats['duration_seconds'] = stats['end_time'] - stats['start_time']

            # Calculate final stats percentages
            total_attempted_analysis = stats['successful_analysis'] + stats['failed_analysis']
            total_for_avg = max(total_attempted_analysis, 1)
            stats['success_rate'] = stats['successful_analysis'] / total_for_avg * 100
            stats['failure_rate'] = stats['failed_analysis'] / total_for_avg * 100
            stats['flagged_rate'] = stats['flagged'] / total_for_avg * 100
            
            # Calculate avg tokens per record
            stats['avg_tokens_per_record'] = stats['grand_total_tokens'] / max(stats['total_processed'], 1)
            
            # Calculate avg execution time
            stats['avg_llm_execution_time'] = stats['total_llm_execution_time'] / max(stats['total_processed'], 1)
            
            # Log final statistics
            logger.info(f"Process completed. Duration: {stats['duration_seconds']:.2f} seconds")
            logger.info(f"Processed {stats['total_processed']} records")
            logger.info(f"Successful analysis: {stats['successful_analysis']} ({stats['success_rate']:.2f}%)")
            logger.info(f"Failed analysis: {stats['failed_analysis']} ({stats['failure_rate']:.2f}%)")
            logger.info(f"Total tokens used: {stats['grand_total_tokens']} (Input: {stats['total_prompt_tokens']}, Output: {stats['total_completion_tokens']})")
            logger.info(f"Average tokens per record: {stats['avg_tokens_per_record']:.2f}")
            logger.info(f"Total LLM execution time: {stats['total_llm_execution_time']:.2f}s")
            logger.info(f"Average LLM execution time per record: {stats['avg_llm_execution_time']:.2f}s")
            
            # Log token counting methods used
            logger.info("Token counting methods used:")
            for method, count in stats['token_counting_methods'].items():
                if count > 0:
                    percentage = (count / stats['total_processed']) * 100 if stats['total_processed'] > 0 else 0
                    logger.info(f"  - {method}: {count} ({percentage:.1f}%)")
            
            # Add token/cost estimation
            estimated_cost = (stats['total_prompt_tokens'] / 1000 * 0.00025) + (stats['total_completion_tokens'] / 1000 * 0.00125)
            logger.info(f"Estimated cost for this run: ${estimated_cost:.4f} (based on Gemini 2.0 Flash pricing)")
            
            return {
                'timestamp': timestamp,
                'total_processed': stats['total_processed'],
                'successful': stats['successful_analysis'],
                'failed': stats['failed_analysis'],
                'tokens': {
                    'input': stats['total_prompt_tokens'],
                    'output': stats['total_completion_tokens'],
                    'total': stats['grand_total_tokens']
                },
                'execution_time': stats['duration_seconds'],
                'llm_execution_time': stats['total_llm_execution_time'],
                'estimated_cost': estimated_cost
            }
            
        except Exception as e:
            logger.error(f"Error in process_all_records: {e}", exc_info=True)
            return None


# Main function to process podcast episodes and identify hosts and guests
def process_summary_host_guest(stop_flag: Optional[threading.Event] = None):
    """
    Process podcast episodes to identify summaries, hosts, and guests.
    
    Args:
        stop_flag: Optional threading.Event that signals when to stop processing
    """
    try:
        # Check if we should stop before starting
        if stop_flag and stop_flag.is_set():
            logger.info("Stop flag set before starting summary_host_guest processing")
            return

        # Run the async process in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_processor():
            try:
                # Initialize the podcast processor
                processor = PodcastProcessor()
                
                # Process all records with Airtable updates
                # Use lower concurrency values to avoid rate limits
                logger.info("Starting podcast analysis with concurrency=2, batch_size=3")
                results = await processor.process_all_records(max_concurrency=2, batch_size=3)
                
                if results:
                    logger.info(f"Process finished successfully. Processed {results['total_processed']} records with {results['successful']} successful.")
                else:
                    logger.info("Process finished, but no results were generated or encountered errors.")
                
                return results
            except Exception as e:
                logger.critical(f"Process failed with unhandled error: {e}", exc_info=True)
                return None
        
        try:
            # Run the processor
            results = loop.run_until_complete(run_processor())
            return results
        finally:
            # Close the loop regardless of outcome
            loop.close()
            
    except Exception as e:
        logger.error(f"Error in process_summary_host_guest: {e}", exc_info=True)
        return None

# This is used when running the script directly
async def main():
    """Main entry point for running the script directly."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Ensure API keys are available
        if not os.getenv("GEMINI_API_KEY"):
            logger.error("GEMINI_API_KEY environment variable is not set. Please check your .env file.")
            return None
            
        # Initialize the podcast processor
        processor = PodcastProcessor()
        
        # Process all records with Airtable updates
        logger.info("Starting podcast analysis with concurrency=2, batch_size=3")
        results = await processor.process_all_records(max_concurrency=2, batch_size=3)
        
        if results:
            logger.info(f"Process finished successfully. Processed {results['total_processed']} records with {results['successful']} successful.")
        else:
            logger.info("Process finished, but no results were generated or encountered errors.")
            
        return results
    except Exception as e:
        logger.critical(f"Main process failed with unhandled error: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    logger.info("=============================================")
    logger.info("Starting Podcast Analysis Process")
    logger.info("=============================================")
    start_run_time = time.time()

    # Call the function using the same interface as the old script
    process_summary_host_guest()

    end_run_time = time.time()
    total_run_duration = end_run_time - start_run_time
    logger.info("=============================================")
    logger.info("Podcast Analysis Process Ended")
    logger.info(f"Total script execution time: {total_run_duration:.2f} seconds")
    logger.info("=============================================")