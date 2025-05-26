"""
OpenAI Service Module

This module defines an OpenAIService class that helps interact with the OpenAI API
to transform unstructured text into structured data, generate chat completions,
and produce genre IDs from user input. The class uses Pydantic models to validate
and structure the data. 

Author: Paschal Okonkwor
Date: 2025-01-06
"""

import os
import json
import logging
import time
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional

from .file_manipulation import read_txt_file
# Import our AI usage tracker
from .ai_usage_tracker import tracker as ai_tracker

# Load .env variables to access your OpenAI API key
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Read in a prompt text from file (example file for genre ID prompt)
category_id_prompt = read_txt_file(
    r"C:\Users\ebube\Documents\PGL transition\app\genre_id_prompt.txt")


class get_host_guest_confirmation(BaseModel):
    status: str = Field(
        description=
        "provide your overall assessment by outputting either 'Both' for if both host and guest are identified, 'Host' or 'Guest'"
    )
    Host: str = Field(
        description=
        "Identify who the host of the show is and extract their names, it is usually one host though"
    )
    Guest: str = Field(
        description=
        "Identify who the guest is and extract their names as a string")


class get_validation(BaseModel):
    correct: str = Field(
        description=
        "the only response I need here is true or false, If the junior correctly labeled the host(s) and guest(s), the boolean variable 'correct' is 'true' or If the junior incorrectly labeled the host(s) and guest(s), the boolean variable 'correct' is 'false' "
    )


class get_answer_for_fit(BaseModel):
    Answer: str = Field(
        description=
        "ONLY provide your overall fit assessment by outputting either 'Fit' or 'Not a fit' (without quotes), using the JSON format specified "
    )


class get_episode_ID(BaseModel):
    ID: str = Field(
        description=
        "Give your output containing ONLY the selected Episode ID in JSON ")


class GetTopicDescriptions(BaseModel):

    topic_1: str = Field(...,
                         alias="Topic 1",
                         description="Title of the first topic.")
    description_1: str = Field(...,
                               alias="Description 1",
                               description="Description for the first topic.")
    topic_2: str = Field(...,
                         alias="Topic 2",
                         description="Title of the second topic.")
    description_2: str = Field(...,
                               alias="Description 2",
                               description="Description for the second topic.")
    topic_3: str = Field(...,
                         alias="Topic 3",
                         description="Title of the third topic.")
    description_3: str = Field(...,
                               alias="Description 3",
                               description="Description for the third topic.")


class getHostName(BaseModel):
    Host: str = Field(
        description=
        "Identify who the host of the show is and extract their names, it is usually one host though, only return the host name nothing more"
    )


class StructuredData(BaseModel):
    """
    This model defines the structure of the data that we expect from our 
    OpenAI completion when generating bios and angles.

    Attributes:
        Bio: A string describing the client's bios (full, summary, short).
        Angles: A string describing the client's angles (topic, outcome, description).
    """
    Bio: str = Field(description="""Client's bio. 
            Include one main text but keep tabs and new lines indicating "Full Bio," 
            "Summary Bio," and "Short Bio" within the text.""")
    Angles: str = Field(description="""Client's angles. 
            Each angle has three parts: Topic, Outcome, Description. 
            Keep tabs and new lines to separate these parts.""")


class OpenAIService:
    """
    A service that communicates with OpenAI's API to perform various text-related tasks.
    It uses 'gpt-4o-2024-08-06' model versions, which might be placeholders in this example.
    """

    def __init__(self):
        """
        Initialize the OpenAI client using the API key from your environment variables.
        """
        try:
            self.client = OpenAI(api_key=os.getenv('OPENAI_API'))
            logger.info("OpenAIService initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    def transform_text_to_structured_data(self, prompt, raw_text, data_type, 
                                         workflow="transform_structured_data", 
                                         podcast_id: Optional[str] = None,
                                         max_retries=3, initial_retry_delay=2):
        """
        Use the OpenAI API to parse raw text into a structured JSON format.

        Args:
            prompt (str): Instructions for what we want from the raw text.
            raw_text (str): The text to be transformed.
            data_type (str): Type of structured data to parse.
            workflow (str): Name of the workflow using this method.
            podcast_id (str, optional): Airtable record ID for tracking purposes.
            max_retries (int): Maximum number of retry attempts
            initial_retry_delay (int): Initial delay in seconds before first retry (doubles with each retry)

        Returns:
            dict: A dictionary that fits the StructuredData model (Bio, Angles).
        """
        retry_count = 0
        retry_delay = initial_retry_delay
        last_exception = None
        
        while retry_count <= max_retries:
            try:
                # Start timing
                start_time = time.time()
                
                # Determine response format based on data_type
                if data_type == 'Structured':
                    response_format = StructuredData
                elif data_type == 'confirmation':
                    response_format = get_host_guest_confirmation
                elif data_type == 'validation':
                    response_format = get_validation
                elif data_type == 'fit':
                    response_format = get_answer_for_fit
                elif data_type == 'episode_ID':
                    response_format = get_episode_ID
                elif data_type == 'topic_descriptions':
                    response_format = GetTopicDescriptions
                elif data_type == 'host_name':
                    response_format = getHostName

                # Set the model name - constant for this method
                model = "gpt-4o-2024-08-06"
                
                # Call the OpenAI API
                completion = self.client.beta.chat.completions.parse(
                    model=model,
                    messages=[{
                        "role":
                        "system",
                        "content":
                        f"{prompt} Please provide the response in JSON format."
                    }, {
                        "role": "user",
                        "content": raw_text
                    }],
                    response_format=response_format,
                )

                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Extract the content
                response_content = completion.choices[0].message.parsed
                
                # Log usage stats from the API response
                tokens_in = completion.usage.prompt_tokens
                tokens_out = completion.usage.completion_tokens
                
                # Log the usage data with podcast_id
                ai_tracker.log_usage(
                    workflow=workflow,
                    model=model,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    execution_time=execution_time,
                    endpoint="openai.beta.chat.completions.parse",
                    podcast_id=podcast_id
                )
                
                logger.debug(f"Parsed structured data: {response_content}")
                return response_content.model_dump()

            except Exception as e:
                last_exception = e
                retry_count += 1
                
                # Log the error with retry information
                if retry_count <= max_retries:
                    logger.warning(f"Error in OpenAI API call (attempt {retry_count}/{max_retries}): {e}. "
                                  f"Retrying in {retry_delay} seconds...")
                    
                    # Wait before retrying with exponential backoff
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Double the delay for next attempt
                else:
                    # We've exhausted all retries
                    logger.error(
                        f"Error during text-to-structured-data transformation after {max_retries} retries: {e}")
                    raise

    def create_chat_completion(self, system_prompt, prompt, 
                              workflow="chat_completion", 
                              podcast_id: Optional[str] = None,
                              parse_json=False, json_key=None,
                              max_retries=3, initial_retry_delay=2):
        """
        Create a chat completion using the OpenAI API. 
        Typically used to generate some JSON-formatted text.

        Args:
            system_prompt (str): The system role prompt for guidance.
            prompt (str): The user's main content/query.
            workflow (str): Name of the workflow using this method.
            podcast_id (str, optional): Airtable record ID for tracking purposes.
            parse_json (bool, optional): Whether to parse the response as JSON.
            json_key (str, optional): If parse_json is True, extract this key from the JSON.
            max_retries (int): Maximum number of retry attempts
            initial_retry_delay (int): Initial delay in seconds before first retry (doubles with each retry)

        Returns:
            str: The raw text (JSON or otherwise) from the assistant's message, or
                 the value of the specified json_key if parse_json is True.
        """
        retry_count = 0
        retry_delay = initial_retry_delay
        last_exception = None
        
        while retry_count <= max_retries:
            try:
                # Start timing
                start_time = time.time()
                
                # Set the model name - constant for this method
                model = "gpt-4o-2024-08-06"
                
                # Call the OpenAI API
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{
                        "role": "system",
                        "content": system_prompt
                    }, {
                        "role": "user",
                        "content": prompt
                    }],
                    temperature=0.1,
                )
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Get the AI's response
                assistant_message = response.choices[0].message.content
                
                # Log usage stats from the API response
                tokens_in = response.usage.prompt_tokens
                tokens_out = response.usage.completion_tokens
                
                # Log the usage data with podcast_id
                ai_tracker.log_usage(
                    workflow=workflow,
                    model=model,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    execution_time=execution_time,
                    endpoint="openai.chat.completions.create",
                    podcast_id=podcast_id
                )
                
                # Parse JSON if requested
                if parse_json:
                    try:
                        # Clean the JSON string in case there are backticks
                        if "```" in assistant_message:
                            assistant_message = assistant_message.split("```")[1]
                            if assistant_message.startswith("json"):
                                assistant_message = assistant_message[4:]
                        
                        result = json.loads(assistant_message.strip())
                        if json_key is not None:
                            if json_key not in result:
                                raise ValueError(f"Response JSON is missing '{json_key}' field.")
                            return result[json_key]
                        return result
                    
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON from response: {e}")
                        logger.error(f"Raw response: {assistant_message}")
                        raise ValueError(
                            f"OpenAI response was not valid JSON. Check logs.")
                
                return assistant_message
                
            except Exception as e:
                last_exception = e
                retry_count += 1
                
                # Log the error with retry information
                if retry_count <= max_retries:
                    logger.warning(f"Error in OpenAI API call (attempt {retry_count}/{max_retries}): {e}. "
                                  f"Retrying in {retry_delay} seconds...")
                    
                    # Wait before retrying with exponential backoff
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Double the delay for next attempt
                else:
                    # We've exhausted all retries
                    logger.error(f"Error in create_chat_completion after {max_retries} retries: {e}")
                    raise Exception(f"Failed to generate chat completion using OpenAI API: {e}")
