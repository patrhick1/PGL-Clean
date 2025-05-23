import os
import json
import logging
import re
import time
import asyncio
from typing import Optional, Union, List, Dict, Any, Tuple
import pandas as pd
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ValidationError
import sys
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain.prompts import PromptTemplate

# Assuming these modules exist and are correctly set up
from airtable_service import PodcastService
# Import the test-specific AI usage tracker
from test_ai_usage_tracker import tracker as test_tracker

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Pydantic models for response validation
class ValidationInfo(BaseModel):
    confidence: float = Field(description="Your confidence level in the host/guest identification, from 0 to 1")
    evidence: str = Field(..., description="Quotes or specific context from the summary that supports your identification")

class TopicPoint(BaseModel):
    name: str = Field(..., description="Name of a main topic discussed in the podcast")
    key_points: List[str] = Field(..., description="List of key points discussed within this topic")

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

# Helper function to run LLM call with retries and parsing
async def _run_llm_analysis(
    llm,
    prompt_template: PromptTemplate,
    parser: PydanticOutputParser,
    episode_title: str,
    text_to_analyze: str,
    record_id: str,
    source_name: str, # e.g., "Summary" or "Transcription"
    model_name: str,  # Added parameter for model name
    test_name: str    # Added parameter for test name
) -> Tuple[Optional[PodcastAnalysis], Dict[str, Any], float]:
    """Runs the LLM analysis for a given text source with retries."""
    analysis_result = None
    token_info = {'input': 0, 'output': 0, 'total': 0}
    start_time = time.time()  # Initialize here to avoid potential reference before assignment
    execution_time = 0

    if not text_to_analyze or len(text_to_analyze.strip()) < 10: # Basic check for meaningful content
        logger.warning(f"Record {record_id}: Skipping analysis for {source_name} due to insufficient content.")
        return None, token_info, execution_time # Return None if no text

    try:
        formatted_prompt = prompt_template.format(
            episode_title=episode_title,
            summary=text_to_analyze # Prompt uses 'summary' variable name, but we pass the text
        )
        logger.debug(f"Record {record_id}: Formatted Prompt for {source_name}:\n{formatted_prompt[:500]}...")

        max_retries = 3 # Reduced from 5 for brevity, adjust as needed
        retry_count = 0
        
        # Longer base delay for all models, with extra delay for Anthropic models
        base_delay = 15 if 'anthropic' in str(llm).lower() else 5

        llm_response_obj = None
        while retry_count < max_retries:
            try:
                # Use the LLM with structured output directly
                llm_with_output = llm.with_structured_output(PodcastAnalysis)
                # Run synchronous LangChain call in a separate thread
                llm_response_obj = await asyncio.to_thread(llm_with_output.invoke, formatted_prompt)

                # Basic validation of the response object
                if not llm_response_obj or not isinstance(llm_response_obj, PodcastAnalysis):
                     raise ValueError("Invalid or empty response structure received from LLM.")

                # --- Token Extraction Logic ---
                try:
                    input_tokens = 0
                    output_tokens = 0
                    total_tokens = 0
                    
                    # Try accessing raw response metadata - safely with try/except
                    try:
                        raw_response = getattr(llm_response_obj, '_raw_response', None) or getattr(llm_response_obj, 'response_metadata', None)
                        usage_metadata = getattr(raw_response, 'usage_metadata', None) if raw_response else None

                        if usage_metadata:
                            input_tokens = usage_metadata.get('prompt_token_count', 0)
                            output_tokens = usage_metadata.get('candidates_token_count', 0)
                            total_tokens = usage_metadata.get('total_token_count', 0)
                    except Exception as e:
                        logger.warning(f"Record {record_id}: Error extracting metadata: {str(e)}")
                    
                    # Fallback if counts are zero but should not be
                    if total_tokens == 0 and input_tokens == 0:
                        input_tokens = len(formatted_prompt) // 4 # Rough estimate
                        try:
                            # Estimate output tokens based on the Pydantic model JSON size
                            output_tokens = len(llm_response_obj.model_dump_json()) // 4
                        except Exception:
                            output_tokens = 200 # Default guess
                        total_tokens = input_tokens + output_tokens
                except Exception as token_error:
                    logger.warning(f"Record {record_id}: Failed to extract token info: {token_error}")
                    # Safe defaults
                    input_tokens = len(formatted_prompt) // 4 
                    output_tokens = 200
                    total_tokens = input_tokens + output_tokens

                token_info = {'input': input_tokens, 'output': output_tokens, 'total': total_tokens}
                # --- End Token Extraction ---

                analysis_result = llm_response_obj
                break # Success

            except (ValidationError, ValueError, AttributeError) as parse_error:
                 # Error during Pydantic parsing or response validation
                 retry_count += 1
                 logger.warning(f"Record {record_id}: Attempt {retry_count}/{max_retries} for {source_name} failed due to parsing/validation error: {parse_error}")
                 if retry_count >= max_retries:
                     logger.error(f"Record {record_id}: Max retries reached for {source_name} after parsing/validation error.")
                     raise parse_error # Re-raise after max retries
                 
                 # Use exponential backoff with jitter also for parsing errors
                 wait_time = base_delay * (2 ** (retry_count - 1)) + random.uniform(0, 2)
                 logger.warning(f"Record {record_id}: Retrying after parsing error in {wait_time:.1f}s...")
                 await asyncio.sleep(wait_time)

            except Exception as e:
                retry_count += 1
                logger.warning(f"Record {record_id}: Attempt {retry_count}/{max_retries} for {source_name} failed: {type(e).__name__} - {e}")
                error_str = str(e).lower()
                is_rate_limit = "quota" in error_str or "429" in error_str or "rate limit" in error_str or "concurrent" in error_str
                is_server_error = "500" in error_str or "503" in error_str or "too many requests" in error_str

                if (is_rate_limit or is_server_error) and retry_count < max_retries:
                    # Double the base delay each retry with more aggressive exponential backoff
                    # Use exponential backoff with jitter to avoid thundering herd problem
                    wait_time = base_delay * (2 ** (retry_count - 1)) + random.uniform(0, 5)
                    
                    # For rate limits specifically, add even more delay
                    if is_rate_limit:
                        wait_time *= 1.5
                        
                    logger.warning(f"Record {record_id}: Rate limit/server error on {source_name}. Retrying in {wait_time:.1f}s...")
                    
                    # For Anthropic rate limits, add an extra message about concurrency
                    if 'anthropic' in str(llm).lower() and ('concurrent' in error_str or 'rate limit' in error_str):
                        logger.warning(f"Record {record_id}: Anthropic rate limit due to concurrent connections - consider using concurrency=1 for Claude models")
                    
                    await asyncio.sleep(wait_time)
                elif retry_count >= max_retries:
                    logger.error(f"Record {record_id}: Max retries reached for {source_name}. Failing analysis.")
                    raise # Re-raise the last exception
                else:
                    # For non-retryable errors, apply shorter delay but still retry
                    wait_time = 2 * (retry_count)
                    logger.error(f"Record {record_id}: Non-rate-limit error. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    if retry_count >= max_retries:
                        raise # Re-raise on final attempt

        execution_time = time.time() - start_time
        logger.info(f"Record {record_id}: Successfully analyzed {source_name}. Time: {execution_time:.2f}s, Tokens: {token_info['total']}")
        
        # Log to test-specific tracker instead of production tracker
        if analysis_result:
            # Map the model parameter to the actual model name for consistent tracking
            actual_model_name = model_name
            # Get actual model name based on provider
            if 'anthropic' in str(llm).lower():
                if 'sonnet' in str(llm.model).lower():
                    actual_model_name = "claude-3-5-sonnet-20241022"
                else:
                    actual_model_name = "claude-3-5-haiku-20241022"
            elif 'genai' in str(llm).lower():
                actual_model_name = "gemini-2.0-flash"
            elif 'openai' in str(llm).lower():
                actual_model_name = "o3-mini"
                
            test_tracker.log_usage(
                workflow="summary_host_guest",
                model=actual_model_name,
                tokens_in=token_info['input'],
                tokens_out=token_info['output'],
                execution_time=execution_time,
                test_name=test_name,
                record_id=record_id
            )
            
        return analysis_result, token_info, execution_time

    except Exception as e:
        logger.error(f"Record {record_id}: Failed to analyze {source_name} after all retries: {type(e).__name__} - {e}", exc_info=True)
        execution_time = time.time() - start_time
        return None, token_info, execution_time # Return None on failure


class PodcastProcessor:
    def __init__(self, model="gemini_flash", test_name="default_test"):
        """
        Initialize with dynamic LLM selection.
        
        Args:
            model (str): Type of LLM to use. Options: "gemini_flash", "claude_haiku", "o3_mini"
            test_name (str): Name to identify this test run for tracking
        """
        try:
            self.airtable_service = PodcastService()
            self.parser = PydanticOutputParser(pydantic_object=PodcastAnalysis)
            self.model_name = model
            self.test_name = test_name
            
            # Set the LLM based on the model parameter
            self.llm = self.select_llm(model)
            
            # Create the prompt template after setting the LLM
            self.prompt_template = self._create_prompt_template()
            
            logger.info(f"Podcast Processor initialized with {model} model for test '{test_name}'")
        except Exception as e:
            logger.critical(f"Failed to initialize PodcastProcessor: {e}", exc_info=True)
            raise
            
    def select_llm(self, model: str):
        """
        Select the LLM based on the model parameter.
        """
        if model.lower() == "claude_haiku":
            api_key = os.getenv("ANTHROPIC_API")
            if not api_key:
                raise ValueError("ANTHROPIC_API environment variable not set")
                
            return ChatAnthropic(
                model="claude-3-5-haiku-20241022", 
                anthropic_api_key=api_key,
                temperature=0.2
            )
            
        elif model.lower() == "claude_sonnet":
            api_key = os.getenv("ANTHROPIC_API")
            if not api_key:
                raise ValueError("ANTHROPIC_API environment variable not set")
                
            return ChatAnthropic(
                model="claude-3-5-sonnet-20241022", 
                anthropic_api_key=api_key,
                temperature=0.3
            )

        elif model.lower() == "gemini_flash":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
                
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", 
                google_api_key=api_key,
                temperature=0.25,
                request_options={"timeout": 180}
            )

        elif model.lower() == "o3_mini":
            api_key = os.getenv("OPENAI_API")
            if not api_key:
                raise ValueError("OPENAI_API environment variable not set")
            
            return ChatOpenAI(
                model_name="o3-mini",
                temperature=None,
                openai_api_key=api_key,
                reasoning_effort="medium"
            )   
        else:
            raise ValueError(f"Unsupported model type: {model}. Use 'gemini_flash', 'claude_haiku', 'claude_sonnet', or 'o3_mini'")

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
Summary: John Pasalis discusses Toronto real estate data with agents Gus Papaioannou and Davin McMahon, and mortgage expert David Larock. Urmi Desai hosts the live chat.
Validation: {{confidence: 0.9, evidence: "Gus Papaioannou and Davin McMahon are identified as 'fellow Realosophy agents' in conversation with the host, and David Larock is identified as a 'Move Smartly mortgage expert' in conversation with the host, indicating they are guests."}}

**Example 2 (Single Guest):**

Provided Text: "Today, John Pasalis interviews real estate expert Sarah Jones about the market trends."
Host: John Pasalis
Guest: Sarah Jones
Status: Both
Summary: John Pasalis interviews Sarah Jones on real estate market trends.
Validation: {{confidence: 0.95, evidence: "Sarah Jones is identified as a 'real estate expert' being interviewed by John Pasalis, indicating she is a guest."}}

**Example 3 (No Guests):**

Provided Text: "John Pasalis discusses the latest market data. Urmi Desai moderates the live chat."
Host: John Pasalis, Urmi Desai
Guest: null
Status: Host
Summary: John Pasalis presents market data, with Urmi Desai moderating the chat.
Validation: {{confidence: 0.85, evidence: "The text only mentions John Pasalis discussing data and Urmi Desai moderating. No other individuals are presented as participating guests in the main discussion."}}

{format_instructions}
""",
                input_variables=["episode_title", "summary"], # 'summary' here matches the template variable name
                partial_variables={"format_instructions": format_instructions}
            )
        except Exception as e:
            logger.error(f"Failed to create prompt template: {e}", exc_info=True)
            raise

    async def process_single_record(self, record_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single podcast record, potentially using Transcription as fallback."""
        record_id = record_data['record_id']
        fields = record_data['fields']
        podcast_id = record_data['podcast_id']

        # Extract potential text sources
        summary_text = fields.get('Summary', '')
        transcription_text = fields.get('Transcription', '') # Get transcription text
        episode_title = fields.get('Episode Title', '')

        # Initialize result data structure with new fields for test tracking
        result_data = {
            'record_id': record_id,
            'episode_title': episode_title,
            'source_used': 'None', # Will be 'Summary', 'Transcription', or 'None'/'Error'
            'ai_summary': None,
            'host': None,
            'guest': None,
            'status': 'None',
            'confidence': 0.0,
            'flagged_human': True,
            'error_reason': '',
            'summary_input_tokens': 0,
            'summary_output_tokens': 0,
            'summary_total_tokens': 0,
            'transcription_input_tokens': 0, # Added field
            'transcription_output_tokens': 0, # Added field
            'transcription_total_tokens': 0,  # Added field
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_tokens': 0,
            'model': self.model_name,  # Add model used
            'test_name': self.test_name,  # Add test name
            'processing_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        summary_analysis: Optional[PodcastAnalysis] = None
        summary_token_info = {'input': 0, 'output': 0, 'total': 0}
        summary_exec_time = 0
        transcription_analysis: Optional[PodcastAnalysis] = None
        transcription_token_info = {'input': 0, 'output': 0, 'total': 0}
        transcription_exec_time = 0

        # --- Step 1: Analyze Summary ---
        logger.info(f"Record {record_id}: Attempting analysis using Summary.")
        if summary_text:
            summary_analysis, summary_token_info, summary_exec_time = await _run_llm_analysis(
                self.llm, self.prompt_template, self.parser, episode_title, summary_text, record_id, 
                "Summary", self.model_name, self.test_name
            )
            result_data['summary_input_tokens'] = summary_token_info['input']
            result_data['summary_output_tokens'] = summary_token_info['output']
            result_data['summary_total_tokens'] = summary_token_info['total']
        else:
            logger.warning(f"Record {record_id}: Summary field is empty. Cannot perform initial analysis.")
            result_data['error_reason'] = "Missing summary"


        # --- Step 2: Decide if Fallback to Transcription is Needed ---
        use_transcription = False
        # FIX: Properly handle the case where summary_analysis might be None
        summary_status = summary_analysis.status if summary_analysis else 'None'
        
        if summary_analysis is None or summary_status != 'Both':
            if transcription_text:
                logger.info(f"Record {record_id}: Summary analysis incomplete (Status: {summary_status}). Attempting fallback using Transcription.")
                use_transcription = True
            else:
                logger.info(f"Record {record_id}: Summary analysis incomplete, but no Transcription text available for fallback.")
        else:
             logger.info(f"Record {record_id}: Summary analysis successful (Status: Both). No fallback needed.")


        # --- Step 3: Analyze Transcription (if needed) ---
        if use_transcription:
            transcription_analysis, transcription_token_info, transcription_exec_time = await _run_llm_analysis(
                self.llm, self.prompt_template, self.parser, episode_title, transcription_text, record_id, 
                "Transcription", self.model_name, self.test_name
            )
            result_data['transcription_input_tokens'] = transcription_token_info['input']
            result_data['transcription_output_tokens'] = transcription_token_info['output']
            result_data['transcription_total_tokens'] = transcription_token_info['total']

        # --- Step 4: Compare Results and Select the Best ---
        final_analysis: Optional[PodcastAnalysis] = None
        source_used = "None"

        # FIX: Better handling of cases where neither analysis succeeded
        if transcription_analysis and summary_analysis:
            # Both analyses ran and succeeded
            summary_is_both = summary_analysis.status == 'Both'
            transcription_is_both = transcription_analysis.status == 'Both'
            summary_confidence = summary_analysis.validation.confidence
            transcription_confidence = transcription_analysis.validation.confidence

            if transcription_is_both and not summary_is_both:
                logger.info(f"Record {record_id}: Choosing Transcription (Status: Both) over Summary (Status: {summary_analysis.status}).")
                final_analysis = transcription_analysis
                source_used = "Transcription"
            elif summary_is_both and not transcription_is_both:
                logger.info(f"Record {record_id}: Choosing Summary (Status: Both) over Transcription (Status: {transcription_analysis.status}).")
                final_analysis = summary_analysis
                source_used = "Summary"
            elif transcription_confidence >= summary_confidence: # Includes Both==Both or Neither==Neither cases, prefer Transcription on tie or higher conf
                logger.info(f"Record {record_id}: Choosing Transcription (Confidence: {transcription_confidence:.2f}) over Summary (Confidence: {summary_confidence:.2f}). Statuses: T='{transcription_analysis.status}', S='{summary_analysis.status}'.")
                final_analysis = transcription_analysis
                source_used = "Transcription"
            else: # Summary confidence is higher
                logger.info(f"Record {record_id}: Choosing Summary (Confidence: {summary_confidence:.2f}) over Transcription (Confidence: {transcription_confidence:.2f}). Statuses: S='{summary_analysis.status}', T='{transcription_analysis.status}'.")
                final_analysis = summary_analysis
                source_used = "Summary"

        elif transcription_analysis: # Only transcription succeeded (or summary was skipped/failed)
            logger.info(f"Record {record_id}: Using Transcription result as Summary analysis was not available or failed.")
            final_analysis = transcription_analysis
            source_used = "Transcription"
        elif summary_analysis: # Only summary succeeded (and transcription wasn't needed or failed)
             logger.info(f"Record {record_id}: Using Summary result. Transcription analysis not performed or failed.")
             final_analysis = summary_analysis
             source_used = "Summary"
        else:
            # Both failed or were skipped
            logger.error(f"Record {record_id}: Both Summary and Transcription analysis failed or were skipped.")
            result_data['error_reason'] = result_data['error_reason'] or "Analysis failed for both Summary and Transcription"
            source_used = "Error"


        # --- Step 5: Populate Final Result Data ---
        result_data['source_used'] = source_used

        if final_analysis:
            host = final_analysis.host
            guest = final_analysis.guest
            status = final_analysis.status
            ai_summary = final_analysis.summary # Use the AI summary from the chosen source
            confidence = final_analysis.validation.confidence

            # FIX: Safer conversion from lists to strings
            if isinstance(host, list): 
                # Filter out None values before converting to string
                host_items = [str(h) for h in host if h is not None]
                host = ", ".join(host_items)
            
            if isinstance(guest, list):
                # Filter out None values before converting to string
                guest_items = [str(g) for g in guest if g is not None]
                guest = ", ".join(guest_items)

            # Determine error reason based on final status/confidence
            error_reason = ''
            flagged = False
            if status == 'Guest':
                error_reason = 'Host not identified'
                flagged = True
            elif status == 'None':
                error_reason = 'Could not identify host or guest'
                flagged = True
            # Flag if confidence is low, even if status is 'Both' or 'Host'
            if confidence < 0.8:
                 # Prepend low confidence warning if another reason already exists
                 error_reason = f"Low confidence identification ({confidence:.2f})" + (f"; {error_reason}" if error_reason else "")
                 flagged = True
            elif status == 'Host' and not guest: # Host identified, no guest found - not an error, potentially not flagged unless confidence is low
                 pass # Not necessarily an error, might not need flagging unless confidence is low (handled above)


            result_data.update({
                'ai_summary': ai_summary,
                'host': host,
                'guest': guest,
                'status': status,
                'confidence': confidence,
                'flagged_human': flagged,
                'error_reason': error_reason,
            })
            logger.info(f"Record {record_id}: Final result - Source: {source_used}, Status: {status}, Confidence: {confidence:.2f}")

        else:
            # Ensure flagged if no analysis succeeded
            result_data['flagged_human'] = True
            if not result_data['error_reason']: # Add generic error if none specific exists
                 result_data['error_reason'] = "Analysis failed to produce results"


        # Update total token counts
        result_data['total_input_tokens'] = result_data['summary_input_tokens'] + result_data['transcription_input_tokens']
        result_data['total_output_tokens'] = result_data['summary_output_tokens'] + result_data['transcription_output_tokens']
        result_data['total_tokens'] = result_data['summary_total_tokens'] + result_data['transcription_total_tokens']

        # Add original summary snippet for context if available
        result_data['summary_snippet'] = summary_text[:500] + "..." if summary_text and len(summary_text) > 500 else summary_text

        return result_data


    # process_batch with better error handling
    async def process_batch(self, batch, semaphore):
        """Process a batch of records with concurrency control and rate limiting."""
        tasks = []
        request_delay = 1 # Seconds between requests within a batch controlled by semaphore
        for record in batch:
            async with semaphore:
                await asyncio.sleep(request_delay)
                # Create a task that handles individual record errors
                async def process_with_error_handling(record_data):
                    try:
                        return await self.process_single_record(record_data)
                    except Exception as e:
                        logger.error(f"Error processing record {record_data.get('record_id', 'unknown')}: {e}", exc_info=True)
                        return {
                            'record_id': record_data.get('record_id', 'unknown'),
                            'error_reason': f"Processing error: {str(e)}",
                            'flagged_human': True,
                            'source_used': 'Error',
                            'status': 'None',
                            'processing_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                
                task = asyncio.create_task(process_with_error_handling(record))
                tasks.append(task)
        
        # Gather results, without return_exceptions=True since we handle them in process_with_error_handling
        return await asyncio.gather(*tasks)


    # process_all_records needs slight modification to handle new token fields in stats/output
    async def process_all_records(self, max_concurrency=3, batch_size=5, record_limit=None):
        """Process podcast records with optional limit."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats = {
            'total_processed': 0,
            'successful_analysis': 0,
            'failed_analysis': 0,
            'used_summary': 0,
            'used_transcription': 0,
            'flagged': 0,
            'total_summary_tokens': 0,
            'total_transcription_tokens': 0,
            'grand_total_tokens': 0,
            'start_time': time.time(),
            'end_time': None,
            'duration_seconds': 0,
            'model': self.model_name,  # Add model information
            'test_name': self.test_name  # Add test name
        }
        results_df = pd.DataFrame()

        try:
            table_name = "Podcast_Episodes"
            view = "No Summary" # Adjust view name if needed
            logger.info(f"Fetching records from Airtable table '{table_name}', view '{view}'...")
            records = self.airtable_service.get_records_from_view(table_name, view)
            logger.info(f"Found {len(records)} record(s) to potentially process.")

            if not records: 
                logger.warning("No records found to process")
                stats['end_time'] = time.time()
                stats['duration_seconds'] = stats['end_time'] - stats['start_time']
                return None

            # Apply limit if specified
            if record_limit and record_limit > 0:
                records = records[:record_limit]
                logger.info(f"Limited to {record_limit} records")

            prepared_records = []
            skipped_preparation = 0
            for record in records[:20]:
                record_id = record['id']
                try:
                    fields = record['fields']
                    podcast_ids = fields.get('Podcast', [])
                    podcast_id = podcast_ids[0] if podcast_ids else None

                    # Skip if missing critical info (Summary OR Transcription must exist)
                    if not podcast_id:
                        logger.warning(f"Skipping record {record_id}: No linked 'Podcast'.")
                        skipped_preparation += 1
                        continue
                    if not fields.get('Summary') and not fields.get('Transcription'):
                         logger.warning(f"Skipping record {record_id}: Both 'Summary' and 'Transcription' fields are empty.")
                         skipped_preparation += 1
                         continue

                    prepared_records.append({
                        'record_id': record_id,
                        'fields': fields,
                        'podcast_id': podcast_id
                    })
                except Exception as e:
                     logger.error(f"Error preparing record {record_id}: {e}", exc_info=True)
                     skipped_preparation += 1

            stats['total_processed'] = skipped_preparation # Start count with skipped ones

            if not prepared_records:
                 logger.warning("No records were successfully prepared for processing.")
                 # Update stats before returning
                 stats['end_time'] = time.time()
                 stats['duration_seconds'] = stats['end_time'] - stats['start_time']
                 # Log final stats here if desired
                 return None

            batches = [prepared_records[i:i + batch_size] for i in range(0, len(prepared_records), batch_size)]
            logger.info(f"Processing {len(prepared_records)} prepared records in {len(batches)} batches...")
            all_results = []
            semaphore = asyncio.Semaphore(max_concurrency)

            for i, batch in enumerate(batches):
                batch_num = i + 1
                logger.info(f"--- Starting Batch {batch_num}/{len(batches)} ({len(batch)} records) ---")
                if i > 0:
                    logger.info(f"Pausing for 2 seconds before batch {batch_num}...")
                    await asyncio.sleep(2)

                start_batch_time = time.time()
                batch_results = await self.process_batch(batch, semaphore)
                batch_duration = time.time() - start_batch_time
                logger.info(f"--- Finished Batch {batch_num}/{len(batches)}. Duration: {batch_duration:.2f}s ---")

                # Process results and update stats
                for result in batch_results:
                    stats['total_processed'] += 1
                    if isinstance(result, Exception):
                        logger.error(f"Batch processing task failed with exception: {result}")
                        stats['failed_analysis'] += 1
                        all_results.append({'record_id': 'UNKNOWN', 'error_reason': f'Batch Task Error: {result}', 'flagged_human': True, 'source_used': 'Error'})
                    elif isinstance(result, dict):
                        all_results.append(result)
                        if result.get('source_used') == 'Error' or not result.get('ai_summary'): # Check if analysis actually failed within the function
                            stats['failed_analysis'] += 1
                            logger.warning(f"Record {result.get('record_id')} analysis failed. Reason: {result.get('error_reason')}")
                        else:
                            stats['successful_analysis'] += 1
                            if result.get('source_used') == 'Summary':
                                stats['used_summary'] += 1
                            elif result.get('source_used') == 'Transcription':
                                stats['used_transcription'] += 1

                        if result.get('flagged_human'): stats['flagged'] += 1

                        # Aggregate token counts
                        stats['total_summary_tokens'] += result.get('summary_total_tokens', 0)
                        stats['total_transcription_tokens'] += result.get('transcription_total_tokens', 0)
                        stats['grand_total_tokens'] += result.get('total_tokens', 0)
                    else:
                         logger.error(f"Unexpected result type from process_batch: {type(result)}")
                         stats['failed_analysis'] += 1 # Count unexpected types as failures

                # Pause logic after every two batches
                if batch_num % 2 == 0 and batch_num < len(batches):
                    pause_duration = 45
                    logger.warning(f"PAUSING for {pause_duration} seconds after processing batch {batch_num}...")
                    await asyncio.sleep(pause_duration)

            # --- Post-processing & Stats ---
            stats['end_time'] = time.time()
            stats['duration_seconds'] = stats['end_time'] - stats['start_time']

            if all_results:
                results_df = pd.DataFrame(all_results)
                # Define column order including new fields
                cols_order = [
                    'record_id', 'episode_title', 'source_used', 'status', 'host', 'guest',
                    'confidence', 'flagged_human', 'error_reason', 'ai_summary',
                    'model', 'test_name',  # Add new metadata columns
                    'summary_snippet',
                    'summary_total_tokens', 'transcription_total_tokens', 'total_tokens',
                    'processing_timestamp'
                    # Add token input/output breakdown if desired
                ]
                for col in cols_order:
                    if col not in results_df.columns: results_df[col] = pd.NA
                results_df = results_df[cols_order]

                # Calculate final stats percentages
                total_attempted_analysis = stats['successful_analysis'] + stats['failed_analysis']
                total_for_avg = max(total_attempted_analysis, 1)
                stats['success_rate'] = stats['successful_analysis'] / total_for_avg * 100
                stats['failure_rate'] = stats['failed_analysis'] / total_for_avg * 100
                stats['flagged_rate'] = stats['flagged'] / total_for_avg * 100
                stats['avg_total_tokens'] = stats['grand_total_tokens'] / total_for_avg

                # Log statistics
                logger.info("--- Processing Statistics ---")
                logger.info(f"  Total records considered: {len(records)}")
                logger.info(f"  Records skipped during preparation: {skipped_preparation}")
                logger.info(f"  Total records attempted analysis: {total_attempted_analysis}")
                logger.info(f"  Successful analysis: {stats['successful_analysis']} ({stats['success_rate']:.1f}%)")
                logger.info(f"  Failed analysis: {stats['failed_analysis']} ({stats['failure_rate']:.1f}%)")
                logger.info(f"    Used Summary source: {stats['used_summary']}")
                logger.info(f"    Used Transcription source: {stats['used_transcription']}")
                logger.info(f"  Records flagged for human review: {stats['flagged']} ({stats['flagged_rate']:.1f}%)")
                logger.info(f"  Grand Total tokens used: {stats['grand_total_tokens']}")
                logger.info(f"    (Summary tokens: {stats['total_summary_tokens']}, Transcription tokens: {stats['total_transcription_tokens']})")
                logger.info(f"  Average total tokens per analyzed record: {stats['avg_total_tokens']:.1f}")
                logger.info(f"  Total processing duration: {stats['duration_seconds']:.2f} seconds")
                logger.info("-----------------------------")

                # Save stats
                stats_file = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    f"summary_host_guest_stats_{self.model_name}_{self.test_name}_{timestamp}.json"
                )
                try:
                    stats_save = stats.copy()
                    stats_save['start_time'] = datetime.fromtimestamp(stats_save['start_time']).isoformat() if stats_save.get('start_time') else None
                    stats_save['end_time'] = datetime.fromtimestamp(stats_save['end_time']).isoformat() if stats_save.get('end_time') else None
                    with open(stats_file, 'w') as f: json.dump(stats_save, f, indent=2)
                    logger.info(f"Processing statistics saved to {stats_file}")
                except Exception as e: logger.error(f"Failed to save statistics to JSON: {e}")

                # Store the results for external access and return them
                self.results = all_results
                return all_results

        except Exception as e:
            logger.critical(f"An critical error occurred in the overall processing: {e}", exc_info=True)
            # Save partial stats on critical failure
            try:
                 if 'duration_seconds' not in stats or stats['duration_seconds'] == 0:
                     stats['end_time'] = time.time()
                     stats['duration_seconds'] = stats['end_time'] - stats.get('start_time', stats['end_time'])
                 stats_file = os.path.join(
                     os.path.dirname(os.path.abspath(__file__)),
                     f"summary_host_guest_stats_{self.model_name}_{self.test_name}_{timestamp}_CRITICAL_FAILED.json"
                 )
                 stats_save = stats.copy()
                 stats_save['start_time'] = datetime.fromtimestamp(stats_save['start_time']).isoformat() if stats_save.get('start_time') else None
                 stats_save['end_time'] = datetime.fromtimestamp(stats_save['end_time']).isoformat() if stats_save.get('end_time') else None
                 with open(stats_file, 'w') as f: json.dump(stats_save, f, indent=2)
                 logger.info(f"Partial processing statistics saved to {stats_file} due to critical error.")
            except Exception as stat_e: logger.error(f"Could not save failure statistics: {stat_e}")
            return results_df if not results_df.empty else None

    def export_results_to_csv(self, output_file: str) -> str:
        """
        Export the results to a CSV file.
        
        Args:
            output_file: Path to the output CSV file
            
        Returns:
            Path to the exported CSV file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use the exact filename format that test_runner.py expects: {workflow}_{model}_{test_name}_{timestamp}.csv
        output_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"summary_host_guest_{self.model_name}_{self.test_name}_{timestamp}.csv"
        )
        
        try:
            # Convert results to DataFrame for easier handling
            df = pd.DataFrame(self.results)
            
            # Export to CSV
            df.to_csv(output_file, index=False)
            logger.info(f"Results exported to {output_file}")
            return output_file
        
        except Exception as e:
            logger.error(f"Failed to export results to CSV: {e}", exc_info=True)
            return None

async def run_test(batch_size: int, concurrency: int, limit: Optional[int], output_file: str, 
                  model: str = "gemini_flash", test_name: str = "default_test",
                  stop_flag: Optional[callable] = None) -> None:
    """
    Run the test processor with the specified parameters.
    
    Args:
        batch_size: Number of records to process in each batch
        concurrency: Maximum number of concurrent processes
        limit: Maximum number of records to process
        output_file: Path to the output CSV file
        model: LLM model to use
        test_name: Name to identify this test run
        stop_flag: Optional callable that returns True when the test should stop
    """
    # Adjust concurrency for Anthropic models due to their rate limits
    if model.lower().startswith("claude"):
        original_concurrency = concurrency
        concurrency = 1  # Force to 1 for Claude models to avoid rate limits
        logger.warning(f"Reducing concurrency from {original_concurrency} to {concurrency} for {model} to avoid rate limits")
        # Also reduce batch size if it's too high
        if batch_size > 2:
            original_batch_size = batch_size
            batch_size = 2
            logger.warning(f"Reducing batch size from {original_batch_size} to {batch_size} for {model} to avoid rate limits")
    
    logger.info(f"Starting test '{test_name}' with batch_size={batch_size}, concurrency={concurrency}, limit={limit}, model={model}")
    
    processor = None
    
    try:
        processor = PodcastProcessor(model=model, test_name=test_name)
        
        # Add stop flag checking if provided
        if stop_flag:
            original_process_batch = processor.process_batch
            
            async def stoppable_process_batch(batch, semaphore):
                if stop_flag():
                    logger.info("Stop flag detected, stopping processing")
                    return []
                return await original_process_batch(batch, semaphore)
            
            # Replace the process_batch method with our stoppable version
            processor.process_batch = stoppable_process_batch
        
        start_time = time.time()
        stats = await processor.process_all_records(
            max_concurrency=concurrency,
            batch_size=batch_size,
            record_limit=limit
        )
        end_time = time.time()
        
        # Only export results if we weren't stopped
        if not (stop_flag and stop_flag()):
            total_duration = end_time - start_time
            logger.info(f"Test '{test_name}' completed in {total_duration:.2f} seconds")
            
            # Export results to CSV
            processor.export_results_to_csv(output_file)
        else:
            logger.info(f"Test '{test_name}' was stopped before completion")
            # Still export any results we have
            if processor and hasattr(processor, 'results') and processor.results:
                logger.info(f"Exporting partial results from stopped test")
                processor.export_results_to_csv(output_file)
        
    except Exception as e:
        logger.critical(f"Error running test: {e}", exc_info=True)
        if processor:
            processor.export_results_to_csv(output_file)
        raise

# Main execution block (assuming main() and if __name__ == "__main__": block are defined as before)
async def main():
    """Main entry point for the script."""
    try:
        import argparse
        parser = argparse.ArgumentParser(description="Test the podcast summary host/guest identification")
        parser.add_argument("--batch-size", type=int, default=5, help="Number of records to process in each batch")
        parser.add_argument("--concurrency", type=int, default=3, help="Maximum number of concurrent processes")
        parser.add_argument("--limit", type=int, default=None, help="Maximum number of records to process")
        parser.add_argument("--output", type=str, default=None, help="Output CSV file path")
        parser.add_argument("--model", type=str, default="gemini_flash", 
                            choices=["gemini_flash", "claude_haiku", "claude_sonnet", "o3_mini"], 
                            help="LLM model to use")
        parser.add_argument("--test-name", type=str, default=None, 
                            help="Name to identify this test run")
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")
        
        args = parser.parse_args()
        
        # Set debug logging if requested
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")
        
        # Set default test name if not specified
        if args.test_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.test_name = f"{args.model}_test_{timestamp}"
            
        # Run test with arguments
        await run_test(
            batch_size=args.batch_size,
            concurrency=args.concurrency,
            limit=args.limit,
            output_file=args.output,
            model=args.model,
            test_name=args.test_name
        )
        
        return None
    except Exception as e:
        logger.critical(f"Main process failed with unhandled error: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv() # Load .env file if it exists

    logger.info("=============================================")
    logger.info("Starting Podcast Analysis Test Process")
    logger.info("=============================================")
    start_run_time = time.time()

    asyncio.run(main())

    end_run_time = time.time()
    total_run_duration = end_run_time - start_run_time
    logger.info("=============================================")
    logger.info(f"Podcast Analysis Test Process Ended")
    logger.info(f"Total script execution time: {total_run_duration:.2f} seconds")
    logger.info("=============================================")