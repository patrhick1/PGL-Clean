# app/anthropic_service.py

import os
from anthropic import Anthropic
from dotenv import load_dotenv
import logging

#Load environment variables
load_dotenv()

# fetch API Key
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API')

# Set up logging
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

class AnthropicService:
    def __init__(self):
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)


    def create_message(self, prompt, model='claude-3-haiku-20240307'):
        try:
            # Call the Claude API
            response = self.client.messages.create(
                model = model,
                max_tokens= 2000,
                messages = [
                    {'role':'user', 'content':f'{prompt}'}
                ],
                temperature= 0.1

            )
            #Return the content
            return response.content[0].text if response.content else "No response received."
        except Exception as e:
            logging.error(f"Error in generate_content_with_claude: {e}")
            raise Exception("Failed to generate message using Anthropic API.") from e