# app/gemini_service.py

import os
import time
import logging
from typing import Optional
from dotenv import load_dotenv
import google.generativeai as genai

# Import our AI usage tracker
from ai_usage_tracker import tracker as ai_tracker

# Load environment variables
load_dotenv()

# Fetch API Key
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class GeminiService:
    """
    A service that communicates with Google's Gemini API to generate text responses.
    """

    def __init__(self):
        """
        Initialize the Gemini client using the API key from environment variables.
        """
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            logger.info("GeminiService initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise

    def create_message(self, prompt, model='gemini-2.0-flash-001', 
                      workflow="unknown", podcast_id: Optional[str] = None,
                      max_retries=3, initial_retry_delay=2):
        """
        Generate a response from Gemini based on the provided prompt.
        
        Args:
            prompt (str): The text prompt to send to the Gemini API
            model (str): The Gemini model to use (default: gemini-1.5-pro)
            workflow (str): The name of the workflow using this method
            podcast_id (str, optional): Airtable record ID for tracking purposes
            max_retries (int): Maximum number of retry attempts
            initial_retry_delay (int): Initial delay in seconds before first retry (doubles with each retry)
            
        Returns:
            str: The text response from Gemini
        """
        retry_count = 0
        retry_delay = initial_retry_delay
        last_exception = None
        
        while retry_count <= max_retries:
            try:
                # Start timing the execution
                start_time = time.time()
                
                # Configure the model
                generation_config = {
                    "temperature": 0.01,
                    "top_p": 0.1,
                    "top_k": 1,
                    "max_output_tokens": 2048,
                }
                
                # Create the model
                model_instance = genai.GenerativeModel(
                    model_name=model,
                    generation_config=generation_config
                )
                
                # Get response from the model
                response = model_instance.generate_content(prompt)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Extract the text content to return
                content_text = response.text if hasattr(response, 'text') else str(response)
                
                # Get token counts if available (counting characters as proxy if not available)
                # Note: the Gemini API might not return token counts directly
                tokens_in = len(prompt) // 4  # Rough approximation
                tokens_out = len(content_text) // 4  # Rough approximation
                
                # Log the API usage with podcast_id
                ai_tracker.log_usage(
                    workflow=workflow,
                    model=model,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    execution_time=execution_time,
                    endpoint="gemini.generate_content",
                    podcast_id=podcast_id
                )
                
                # Return the content
                return content_text
                
            except Exception as e:
                last_exception = e
                retry_count += 1
                
                # Log the error with retry information
                if retry_count <= max_retries:
                    logger.warning(f"Error in Gemini API call (attempt {retry_count}/{max_retries}): {e}. "
                                  f"Retrying in {retry_delay} seconds...")
                    
                    # Wait before retrying with exponential backoff
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Double the delay for next attempt
                else:
                    # We've exhausted all retries
                    logger.error(f"Error in create_message after {max_retries} retries: {e}")
                    raise Exception(
                        "Failed to generate message using Gemini API.") from e
    
    def create_chat_completion(self, system_prompt, prompt, 
                              workflow="chat_completion", 
                              podcast_id: Optional[str] = None,
                              max_retries=3, initial_retry_delay=2):
        """
        Create a chat completion using the Gemini API.
        
        Args:
            system_prompt (str): The system role prompt for guidance.
            prompt (str): The user's main content/query.
            workflow (str): Name of the workflow using this method.
            podcast_id (str, optional): Airtable record ID for tracking purposes.
            max_retries (int): Maximum number of retry attempts
            initial_retry_delay (int): Initial delay in seconds before first retry (doubles with each retry)
            
        Returns:
            str: The response text from Gemini
        """
        # Combine system prompt and user prompt for Gemini
        combined_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
        
        # Get the raw response from Gemini
        return self.create_message(
            prompt=combined_prompt,
            model="gemini-2.0-pro-exp-02-05",
            workflow=workflow,
            podcast_id=podcast_id,
            max_retries=max_retries,
            initial_retry_delay=initial_retry_delay
        )


# Test the service if run directly
if __name__ == "__main__":
    gemini_service = GeminiService()
    
    # Test a simple query
    print("Testing simple message generation:")
    response = gemini_service.create_message(
        "Write a one-paragraph summary about podcasting.",
        workflow="test"
    )
    print(response)
    print("\n" + "-"*50 + "\n")
    
    # Test chat completion
    print("Testing chat completion:")
    system_prompt = "You are a helpful AI assistant specializing in podcast information."
    user_prompt = "What are three popular podcast hosting platforms?"
    response = gemini_service.create_chat_completion(
        system_prompt=system_prompt,
        prompt=user_prompt,
        workflow="test"
    )
    print(response) 