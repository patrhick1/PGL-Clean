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
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

from file_manipulation import read_txt_file

# Load .env variables to access your OpenAI API key
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Read in a prompt text from file (example file for genre ID prompt)
genre_id_prompt = read_txt_file(r"C:\Users\ebube\Documents\PGL transition\app\genre_id_prompt.txt")

class StructuredData(BaseModel):
    """
    This model defines the structure of the data that we expect from our 
    OpenAI completion when generating bios and angles.

    Attributes:
        Bio: A string describing the client's bios (full, summary, short).
        Angles: A string describing the client's angles (topic, outcome, description).
    """
    Bio: str = Field(
        description="""Client's bio. 
            Include one main text but keep tabs and new lines indicating "Full Bio," 
            "Summary Bio," and "Short Bio" within the text."""
    )
    Angles: str = Field(
        description="""Client's angles. 
            Each angle has three parts: Topic, Outcome, Description. 
            Keep tabs and new lines to separate these parts."""
    )

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

    def transform_text_to_structured_data(self, prompt, raw_text):
        """
        Use the OpenAI API to parse raw text into a structured JSON format.

        Args:
            prompt (str): Instructions for what we want from the raw text.
            raw_text (str): The text to be transformed.

        Returns:
            dict: A dictionary that fits the StructuredData model (Bio, Angles).
        """
        try:
            # Beta usage: parse is a hypothetical or specialized method 
            # for this version of the OpenAI Python library
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {
                        "role": "system",
                        "content": f"{prompt} Please provide the response in JSON format."
                    },
                    {
                        "role": "user",
                        "content": raw_text
                    }
                ],
                response_format=StructuredData,
            )

            response_content = completion.choices[0].message.parsed
            logger.debug(f"Parsed structured data: {response_content}")
            return response_content.model_dump()

        except Exception as e:
            logger.error(f"Error during text-to-structured-data transformation: {e}")
            raise

    def create_chat_completion(self, system_prompt, prompt):
        """
        Create a chat completion using the OpenAI API. 
        Typically used to generate some JSON-formatted text.

        Args:
            system_prompt (str): The system role prompt for guidance.
            prompt (str): The user's main content/query.

        Returns:
            str: The raw text (JSON or otherwise) from the assistant's message.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {
                        "role": "system",
                        "content": f"{system_prompt} Please provide the response in JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=2000,
                temperature=1,
            )
            assistant_message = response.choices[0].message.content
            logger.debug(f"Chat completion response: {assistant_message}")
            return assistant_message

        except Exception as e:
            logger.error(f"Error creating chat completion: {e}")
            raise

    def generate_genre_ids(self, run_keyword):
        """
        Generate a list of genre IDs based on a user keyword. 
        The final response is expected to be JSON containing an 'ids' field.

        Example JSON Format:
            {"ids": "139,144,157,99"}

        Args:
            run_keyword (str): The keyword or phrase to generate genre IDs from.

        Returns:
            str: A comma-separated string of genre IDs.

        Raises:
            ValueError: If the assistant's reply is not valid JSON or doesn't have 'ids'.
        """
        prompt = f"""
        User Search Query:
        "{run_keyword}"

        Provide the list of genre IDs as per the example above. 
        Return the response in JSON format with an 'ids' key containing an array of integers.
        Do not include backticks i.e ```json
        Example JSON Output Format: {{"ids": "139,144,157,99,90,77,253,69,104,84"}}
        """

        try:
            response = self.client.chat.completions.create(
                model='gpt-4o-2024-08-06',
                messages=[
                    {'role': 'system', 'content': genre_id_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                max_tokens=2048,
                n=1,
                stop=None,
                temperature=0.5,
            )

            assistant_message = response.choices[0].message.content
            logger.debug(f"Genre ID assistant message: {assistant_message}")

            # Attempt to parse JSON
            result = json.loads(assistant_message)
            genre_ids = result['ids']
            return genre_ids

        except Exception as e:
            logger.error(f"Error parsing genre IDs from OpenAI response: {e}")
            raise ValueError(f"Error parsing OpenAI response: {e}")
