# app/anthropic_service.py

import os
import time
from anthropic import Anthropic
from dotenv import load_dotenv
import logging
from typing import Optional

# Import our AI usage tracker
from .ai_usage_tracker import tracker as ai_tracker

#Load environment variables
load_dotenv()

# fetch API Key
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API')

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


class AnthropicService:

    def __init__(self):
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)

    def create_message(self, prompt, model='claude-3-5-haiku-20241022', 
                      workflow="unknown", podcast_id: Optional[str] = None,
                      max_retries=3, initial_retry_delay=2):
        """
        Generate a response from Claude based on the provided prompt.
        
        Args:
            prompt (str): The text prompt to send to the Claude API
            model (str): The Claude model to use
            workflow (str): The name of the workflow using this method
            podcast_id (str, optional): Airtable record ID for tracking purposes
            max_retries (int): Maximum number of retry attempts
            initial_retry_delay (int): Initial delay in seconds before first retry (doubles with each retry)
            
        Returns:
            str: The text response from Claude
        """
        retry_count = 0
        retry_delay = initial_retry_delay
        last_exception = None
        
        while retry_count <= max_retries:
            try:
                # Start timing the execution
                start_time = time.time()
                
                # Call the Claude API
                response = self.client.messages.create(model=model,
                                                   max_tokens=2000,
                                                   messages=[{
                                                       'role': 'user',
                                                       'content': f'{prompt}'
                                                   }],
                                                   temperature=0.1)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Extract the text content to return
                content_text = response.content[0].text if response.content else "No response received."
                
                # Track token usage
                tokens_in = response.usage.input_tokens
                tokens_out = response.usage.output_tokens
                
                # Log the API usage with podcast_id
                ai_tracker.log_usage(
                    workflow=workflow,
                    model=model,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    execution_time=execution_time,
                    endpoint="anthropic.messages.create",
                    podcast_id=podcast_id
                )
                
                # Return the content
                return content_text
                
            except Exception as e:
                last_exception = e
                retry_count += 1
                
                # Log the error with retry information
                if retry_count <= max_retries:
                    logging.warning(f"Error in Anthropic API call (attempt {retry_count}/{max_retries}): {e}. "
                                   f"Retrying in {retry_delay} seconds...")
                    
                    # Wait before retrying with exponential backoff
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Double the delay for next attempt
                else:
                    # We've exhausted all retries
                    logging.error(f"Error in generate_content_with_claude after {max_retries} retries: {e}")
                    raise Exception(
                        "Failed to generate message using Anthropic API.") from e
