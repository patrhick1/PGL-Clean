"""
Angles Generation

This module creates angles (topics and descriptions) for a client's bio in different
versions (v1, v2) based on information stored in Google Docs and Airtable. 
It uses external services like Gemini, OpenAI and Google Docs to analyze content.

Author: Paschal Okonkwor
Date: 2025-01-06
"""

import logging
import asyncio
import concurrent.futures
from typing import Dict, Any, Optional, List, Tuple
import time
from datetime import datetime
import os
import random
import backoff  # If you don't have this, you'll need to install it
import json
import traceback

# Import LangChain components for Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
# Add LangChain text splitting imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import SentenceTransformersTokenTextSplitter

from .airtable_service import MIPRService
from .google_docs_service import GoogleDocsService
from .openai_service import OpenAIService
# Remove GeminiService import and use LangChain instead
from .data_processor import extract_document_id
from .ai_usage_tracker import tracker as ai_tracker

# Enhanced logging for troubleshooting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnglesProcessor:
    """
    Class to process and generate angles and bios based on client information.
    Uses a more efficient approach with content summarization and Gemini 2.0 Flash model.
    """
    
    def __init__(self):
        """Initialize services needed for angles processing."""
        # Print service account info (without revealing credentials)
        service_account_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if service_account_path:
            logger.info(f"Using service account file: {service_account_path}")
            if os.path.exists(service_account_path):
                logger.info("Service account file exists")
                try:
                    with open(service_account_path, 'r') as f:
                        service_info = json.load(f)
                        # Log just enough info to verify it's correct without revealing secrets
                        logger.info(f"Service account client_email: {service_info.get('client_email')}")
                        logger.info(f"Service account project_id: {service_info.get('project_id')}")
                except Exception as e:
                    logger.error(f"Error reading service account file: {e}")
            else:
                logger.error(f"Service account file does not exist: {service_account_path}")
        else:
            logger.error("GOOGLE_APPLICATION_CREDENTIALS not set in environment")
        
        self.airtable_service = MIPRService()
        self.google_docs_service = GoogleDocsService()
        self.openai_service = OpenAIService()
        
        # Initialize the LangChain Gemini model
        self.gemini_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            google_api_key=os.getenv("GEMINI_API_KEY"), 
            temperature=0.3,
            max_output_tokens=None  # Set to None to avoid output token limits
        )
        
        # Create a thread pool executor for running synchronous API calls
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        
        # Set prompt document IDs as class attributes
        self.prompt_doc_id_v1 = '1r8FUzNCWkJRdBpe87diiP645X1uOC6GJKdyGEK6s_Qs'
        self.keyword_prompt_doc_id = '18r8jTqj5cCzhnlajjKTPCJ4roG7kxTTNgCGAlW5WxoM'
        self.prompt_doc_id_v2 = '1hk3sietKNY29wrq9_O5iLJ1Any_lmQ8FSNY6mAY5OG8'
        
        # Test if we can access the prompt documents
        logger.info("Testing access to prompt documents in constructor:")
        try:
            # This is synchronous, so it should work if permissions are correct
            prompt_v1 = self.google_docs_service.get_document_content(self.prompt_doc_id_v1)
            logger.info(f"Successfully retrieved prompt_v1 document: {len(prompt_v1)} characters")
        except Exception as e:
            logger.error(f"Error accessing prompt_v1 document in constructor: {e}")
            logger.error(traceback.format_exc())
        
        # Tracking stats
        self.stats = {
            'records_processed': 0,
            'successful': 0,
            'failed': 0,
            'token_usage': {'input': 0, 'output': 0},
            'execution_times': []
        }

    async def generate_text_with_gemini(self, prompt: str, workflow: str = "bio_and_angles", delay: float = 1.0) -> str:
        """
        Generate text using the LangChain Gemini model.
        
        Args:
            prompt: The text prompt to send to Gemini
            workflow: The workflow name for usage tracking
            delay: Time in seconds to wait before making the API call (to avoid rate limits)
            
        Returns:
            Generated text as string
        """
        start_time = time.time()
        
        # Add a delay before making the API call to avoid rate limits
        await asyncio.sleep(delay)
        
        try:
            # Use run_in_executor since LangChain's call is synchronous
            loop = asyncio.get_running_loop()
            
            # Create the message content for LangChain
            messages = [HumanMessage(content=prompt)]
            
            # Execute the LangChain call in a thread to not block the async loop
            response = await loop.run_in_executor(
                self.executor,
                lambda: self.gemini_model.invoke(messages)
            )
            
            # Extract text from the response
            text_response = response.content
            
            # Track usage
            execution_time = time.time() - start_time
            tokens_in = len(prompt) // 4  # Rough approximation
            tokens_out = len(text_response) // 4  # Rough approximation
            
            # Log the API usage
            ai_tracker.log_usage(
                workflow=workflow,
                model="gemini-2.0-flash",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                execution_time=execution_time,
                endpoint="gemini.generate_content"
            )
            
            # Update local stats
            self.stats['token_usage']['input'] += tokens_in
            self.stats['token_usage']['output'] += tokens_out
            
            return text_response
            
        except Exception as e:
            logger.error(f"Error generating text with Gemini: {e}")
            
            # If we hit a rate limit error, retry with a longer delay
            if "429" in str(e) or "quota" in str(e).lower() or "rate limit" in str(e).lower():
                retry_delay = delay * 2  # Exponential backoff
                if retry_delay > 30:  # Cap at 30 seconds
                    retry_delay = 30
                    
                logger.info(f"Hit rate limit, retrying with {retry_delay}s delay")
                await asyncio.sleep(retry_delay)
                return await self.generate_text_with_gemini(prompt, workflow, retry_delay)
            
            raise

    async def summarize_content(self, content: str, title: str) -> str:
        """
        Summarize content to reduce token usage in main prompt.
        Uses LangChain text splitting for semantic chunking with overlap.
        
        Args:
            content: The text content to summarize
            title: Title describing the content for better context
            
        Returns:
            Summarized version of the content
        """
        if not content.strip():
            return ""
            
        # Skip summarization for very short content
        if len(content) < 50000:
            return content
            
        try:
            # Initialize a semantic text splitter with overlap
            # First split by semantic units (paragraphs, sentences)
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ". ", " ", ""],
                chunk_size=50000,  # Smaller chunk size for better semantic coherence
                chunk_overlap=5000,  # Add overlap to maintain context between chunks
                length_function=len,
            )
            
            # Split the text into semantic chunks
            chunks = text_splitter.split_text(content)
            logger.info(f"Split {title} content into {len(chunks)} semantic chunks")
            
            # Calculate base delay based on content length to avoid rate limits
            base_delay = 1.0
            content_length_factor = min(len(content) / 10000, 5)  # Cap at 5x
            adaptive_delay = base_delay * (1 + content_length_factor)
            
            logger.info(f"Using {adaptive_delay:.2f}s base delay for summarizing {title} ({len(content)} chars)")
            
            summary = ""
            
            # Process each chunk with increasing delays to avoid rate limits
            for i, chunk in enumerate(chunks):
                # Skip empty chunks
                if not chunk.strip():
                    continue
                    
                # Use increasing delays for subsequent chunks
                chunk_delay = adaptive_delay * (1 + i * 0.3)  # Increase delay by 30% for each chunk
                
                # Create a prompt for this chunk
                if i == 0:
                    chunk_prompt = f"""
                    Summarize the following {title} while preserving all important points, topics, 
                    opinions, and unique perspectives. Focus on key themes, expertise areas, 
                    and memorable quotes. Be concise but comprehensive.
                    
                    Content:
                    {chunk}
                    """
                else:
                    chunk_prompt = f"""
                    Continue summarizing this {title} content while preserving all important points, 
                    topics, opinions, and unique perspectives. This is part {i+1} of the content.
                    
                    Content:
                    {chunk}
                    """
                
                logger.info(f"Processing chunk {i+1}/{len(chunks)} with {chunk_delay:.2f}s delay")
                
                # Use the LangChain-based Gemini method with increasing delays
                chunk_summary = await self.generate_text_with_gemini(
                    prompt=chunk_prompt,
                    workflow="bio_and_angles",
                    delay=chunk_delay
                )
                
                # Add a separator between chunk summaries for clarity
                if summary and chunk_summary:
                    summary += "\n\n" + chunk_summary
                else:
                    summary += chunk_summary
            
            logger.info(f"Successfully summarized {title} content: {len(content)} chars → {len(summary)} chars")
            return summary
            
        except Exception as e:
            logger.warning(f"Error summarizing {title} content: {e}. Using original content.")
            # On error, return a truncated version of the original to ensure we don't exceed token limits
            return content[:3000]

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=60,
        on_backoff=lambda details: logger.info(f"Retrying document retrieval after {details['wait']:.1f}s")
    )
    async def get_document_content_async(self, doc_id: Optional[str]) -> str:
        """
        Retrieve document content asynchronously using run_in_executor.
        Added retry logic with exponential backoff.
        
        Args:
            doc_id: The Google Document ID
            
        Returns:
            Document content as string
        """
        if not doc_id:
            return ""
            
        loop = asyncio.get_running_loop()
        
        # First, log that we're attempting to access the document
        logger.info(f"Attempting to retrieve document {doc_id} asynchronously")
        
        # Store the number of attempts for debugging
        attempt = 0
        
        try:
            # Add a small random delay to prevent rate limiting
            await asyncio.sleep(random.uniform(0.5, 2.0))
            attempt += 1
            
            async_start = time.time()
            logger.info(f"Starting async document retrieval for {doc_id} (attempt {attempt})")
            
            content = await loop.run_in_executor(
                self.executor,
                lambda: self.google_docs_service.get_document_content(doc_id)
            )
            
            async_duration = time.time() - async_start
            logger.info(f"Async document retrieval took {async_duration:.2f}s for {doc_id}")
            
            if not content:
                logger.warning(f"Document {doc_id} returned empty content")
            
            return content
            
        except Exception as e:
            # Enhanced error logging
            logger.error(f"Error retrieving document {doc_id} asynchronously: {e}")
            logger.error(traceback.format_exc())
            
            # If we've reached max retries and still failing, try a direct synchronous approach
            logger.info(f"Attempting direct synchronous retrieval for document {doc_id} as fallback")
            try:
                # Try to access the document directly without the executor
                sync_start = time.time()
                content = self.google_docs_service.get_document_content(doc_id)
                sync_duration = time.time() - sync_start
                logger.info(f"Synchronous document retrieval succeeded in {sync_duration:.2f}s for {doc_id}")
                return content
            except Exception as sync_e:
                logger.error(f"Synchronous document retrieval also failed for {doc_id}: {sync_e}")
                # Re-raise to trigger backoff retry or eventually fail
                raise

    async def get_content_from_docs(self, record: Dict[str, Any]) -> Tuple[str, str, str]:
        """
        Retrieve and summarize content from Google Docs links.
        
        Args:
            record: Airtable record dictionary
            
        Returns:
            Tuple of summarized (social_content, podcast_content, articles_content)
        """
        fields = record.get('fields', {})
        name = fields.get('Name', '')
        
        # Get document links
        social_media_posts_link = fields.get('Social Media posts', '')
        podcast_transcripts_link = fields.get('Podcast transcripts', '')
        articles_link = fields.get('Articles', '')
        
        # Extract document IDs
        social_doc_id = extract_document_id(social_media_posts_link)
        podcast_doc_id = extract_document_id(podcast_transcripts_link)
        articles_doc_id = extract_document_id(articles_link)
        
        # Log the document IDs we're about to retrieve
        logger.info(f"Retrieving documents for {name}: Social={social_doc_id}, Podcast={podcast_doc_id}, Articles={articles_doc_id}")
        
        # Get content from Google Docs asynchronously with individual error handling
        social_content = ""
        podcast_content = ""
        articles_content = ""
        
        try:
            if social_doc_id:
                social_content = await self.get_document_content_async(social_doc_id)
        except Exception as e:
            logger.error(f"Failed to retrieve social content: {e}")
            
        try:
            if podcast_doc_id:
                podcast_content = await self.get_document_content_async(podcast_doc_id)
        except Exception as e:
            logger.error(f"Failed to retrieve podcast content: {e}")
            
        try:
            if articles_doc_id:
                articles_content = await self.get_document_content_async(articles_doc_id)
        except Exception as e:
            logger.error(f"Failed to retrieve articles content: {e}")
        
        # Summarize content asynchronously
        summarization_tasks = [
            self.summarize_content(social_content, f"{name}'s Social Media Posts"),
            self.summarize_content(podcast_content, f"{name}'s Podcast Transcripts"),
            self.summarize_content(articles_content, f"{name}'s Articles")
        ]
        
        summarized_content = await asyncio.gather(*summarization_tasks)
        return summarized_content

    async def generate_structured_data(self, name: str, content: Tuple[str, str, str], prompt_template: str, version: str) -> Dict[str, Any]:
        """
        Generate structured data for bios and angles using Gemini.
        
        Args:
            name: Client name
            content: Tuple of (social_content, podcast_content, articles_content)
            prompt_template: Template for the prompt
            version: Version string ('v1' or 'v2')
            
        Returns:
            Dictionary containing structured bio and angles data
        """
        social_content, podcast_content, articles_content = content
        
        # Prepare the prompt for Gemini
        prompt = f"""
        Client Name: {name}

        Social Media posts:
        {social_content}

        Podcast transcripts:
        {podcast_content}

        Articles:
        {articles_content}

        {prompt_template}

        Create the bios and angles for {name}
        """
        
        # Calculate content size for adaptive delay
        total_content_length = len(social_content) + len(podcast_content) + len(articles_content) + len(prompt_template)
        base_delay = 2.0  # Base delay in seconds
        content_length_factor = min(total_content_length / 50000, 4)  # Cap at 4x
        adaptive_delay = base_delay * (1 + content_length_factor)
        
        logger.info(f"Generating bios and angles with {adaptive_delay:.2f}s delay (content size: {total_content_length} chars)")
        
        # Call Gemini to generate the text using LangChain with adaptive delay
        gemini_response = await self.generate_text_with_gemini(
            prompt=prompt,
            workflow="bio_and_angles",
            delay=adaptive_delay
        )
        
        logger.info(f"Gemini response received: {len(gemini_response)} characters. Parsing with OpenAI...")
        
        # Add a brief delay before the OpenAI call to spread out API usage
        await asyncio.sleep(1.5)
        
        # Use OpenAI to structure the text (this could also be replaced with Gemini)
        loop = asyncio.get_running_loop()
        structured_data = await loop.run_in_executor(
            self.executor,
            lambda: self.openai_service.transform_text_to_structured_data(
                prompt="Parse out each of 3 bios and client angles, each angle has 3 parts: Topic, Outcome, Description", 
                raw_text=gemini_response,
                data_type="Structured",
                workflow="bio_and_angles"
            )
        )
        
        return structured_data

    async def create_document_async(self, title: str, content: str) -> str:
        """
        Create a Google Doc asynchronously.
        
        Args:
            title: Document title
            content: Document content
            
        Returns:
            Document link
        """
        loop = asyncio.get_running_loop()
        doc_link = await loop.run_in_executor(
            self.executor,
            lambda: self.google_docs_service.create_document(title, content)
        )
        return doc_link

    async def share_document_async(self, doc_id: str):
        """
        Share a Google Doc asynchronously.
        
        Args:
            doc_id: Document ID
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self.executor,
            lambda: self.google_docs_service.share_document(doc_id)
        )

    async def create_documents_and_update(self, 
                                         record_id: str, 
                                         name: str, 
                                         structured_data: Dict[str, Any], 
                                         version: str) -> Dict[str, Any]:
        """
        Create Google Docs for the generated bios and angles and update Airtable.
        
        Args:
            record_id: Airtable record ID
            name: Client name
            structured_data: Dictionary containing structured bio and angles data
            version: Version string ('v1' or 'v2')
            
        Returns:
            Dictionary containing document links and status
        """
        logger.info(f"Creating documents for {name} ({version})")
        
        # Add delay before creating documents to avoid Google API rate limits
        await asyncio.sleep(2.0)
        
        # Create new Google Docs for Bio and Angles asynchronously
        bio_doc_link = await self.create_document_async(
            f'{name} - Bio {version}', 
            structured_data.get("Bio")
        )
        
        bio_doc_id = bio_doc_link.split('/')[-2]
        logger.info(f"Created Bio document: {bio_doc_link}")
        
        # Add delay between document creation to avoid rate limits
        await asyncio.sleep(3.0)
        
        await self.share_document_async(bio_doc_id)
        logger.info(f"Shared Bio document with ID: {bio_doc_id}")
        
        # Add delay before creating the next document
        await asyncio.sleep(3.0)
        
        angles_doc_link = await self.create_document_async(
            f'{name} - Angles {version}', 
            structured_data.get("Angles")
        )
        
        angles_doc_id = angles_doc_link.split('/')[-2]
        logger.info(f"Created Angles document: {angles_doc_link}")
        
        # Add delay between document operations
        await asyncio.sleep(2.0)
        
        await self.share_document_async(angles_doc_id)
        logger.info(f"Shared Angles document with ID: {angles_doc_id}")
        
        # Generate keywords if version is v1
        keywords = ""
        if version == "v1":
            # Add delay before keyword generation
            await asyncio.sleep(2.0)
            
            # Get keyword prompt template
            keyword_prompt_content = await self.get_document_content_async(self.keyword_prompt_doc_id)
            
            # Generate keywords using OpenAI
            prompt_for_keywords = f"""
                Bio: 
                {structured_data.get("Bio")}

                Angles:
                {structured_data.get("Angles")}
            """
            
            logger.info("Generating keywords using OpenAI")
            loop = asyncio.get_running_loop()
            keywords = await loop.run_in_executor(
                self.executor,
                lambda: self.openai_service.create_chat_completion(
                    keyword_prompt_content, 
                    prompt_for_keywords,
                    workflow="bio_and_angles"
                )
            )
            logger.info(f"Generated {len(keywords)} characters of keywords")
        
        # Add delay before updating Airtable
        await asyncio.sleep(2.0)
        
        # Update the Airtable record
        update_fields = {
            f'Bio {version}': bio_doc_link,
            f'Angles {version}': angles_doc_link,
            'Angles & Bio Button': False  # Reset the button
        }
        
        # Add keywords if present
        if keywords:
            update_fields['Keywords'] = keywords
        
        # Update Airtable asynchronously    
        loop = asyncio.get_running_loop()
        logger.info(f"Updating Airtable record {record_id} with document links")
        await loop.run_in_executor(
            self.executor,
            lambda: self.airtable_service.update_record(record_id, update_fields)
        )
        logger.info("Airtable record updated successfully")
        
        return {
            'bio_link': bio_doc_link,
            'angles_link': angles_doc_link,
            'keywords': keywords if version == "v1" else None,
            'status': 'success'
        }

    async def process_record_v1(self, record_id: str) -> Dict[str, Any]:
        """
        Process a single record to generate v1 bio and angles.
        
        Args:
            record_id: Airtable record ID
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        try:
            # Retrieve the record from Airtable
            loop = asyncio.get_running_loop()
            record = await loop.run_in_executor(
                self.executor,
                lambda: self.airtable_service.get_record_by_id(record_id)
            )
            
            fields = record.get('fields', {})
            
            # Check if 'Angles & Bio Button' is true
            if not fields.get('Angles & Bio Button'):
                logger.info(f"Record {record_id} does not have 'Angles & Bio Button' enabled.")
                return {'status': 'skipped', 'reason': 'Angles & Bio Button not enabled'}
            
            name = fields.get('Name', '')
            logger.info(f"Processing record for {name} (v1)")
            
            # Get prompt template
            prompt_content = await self.get_document_content_async(self.prompt_doc_id_v1)
            
            # Get content from Google Docs
            content = await self.get_content_from_docs(record)
            
            # Generate structured data
            structured_data = await self.generate_structured_data(name, content, prompt_content, "v1")
            
            # Create documents and update Airtable
            result = await self.create_documents_and_update(record_id, name, structured_data, "v1")
            
            self.stats['successful'] += 1
            execution_time = time.time() - start_time
            self.stats['execution_times'].append(execution_time)
            
            logger.info(f"Successfully generated bios and angles for {name} (v1) in {execution_time:.2f} seconds")
            
            return {**result, 'execution_time': execution_time}
            
        except Exception as e:
            self.stats['failed'] += 1
            execution_time = time.time() - start_time
            logger.error(f"Error generating Angles and Bio v1 for record {record_id}: {e}")
            return {'status': 'error', 'error': str(e), 'execution_time': execution_time}
        finally:
            self.stats['records_processed'] += 1

    async def process_record_v2(self, record_id: str) -> Dict[str, Any]:
        """
        Process a single record to generate v2 bio and angles.
        
        Args:
            record_id: Airtable record ID
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        try:
            # Retrieve the record from Airtable
            loop = asyncio.get_running_loop()
            record = await loop.run_in_executor(
                self.executor,
                lambda: self.airtable_service.get_record_by_id(record_id)
            )
            
            fields = record.get('fields', {})
            
            # Check if 'Angles & Bio Button' and 'Mock Interview Email Send' are true
            has_bio_button = fields.get('Angles & Bio Button')
            has_mock_send = fields.get('Mock Interview Email Send')
            
            if not (has_bio_button and has_mock_send):
                logger.info(f"Record {record_id} is missing required fields for v2.")
                return {'status': 'skipped', 'reason': 'Missing required fields for v2'}
            
            name = fields.get('Name', '')
            logger.info(f"Processing record for {name} (v2)")
            
            # Get prompt template
            prompt_content = await self.get_document_content_async(self.prompt_doc_id_v2)
            
            # For v2, we need the v1 content and transcript
            bio_link = fields.get('Bio v1', '')
            angles_link = fields.get('Angles v1', '')
            transcript_link = fields.get('Transcription with client', '')
            
            # Extract document IDs
            bio_doc_id = extract_document_id(bio_link)
            angles_doc_id = extract_document_id(angles_link)
            transcript_doc_id = extract_document_id(transcript_link)
            
            # Fetch content from Google Docs
            docs_tasks = [
                self.get_document_content_async(bio_doc_id),
                self.get_document_content_async(angles_doc_id),
                self.get_document_content_async(transcript_doc_id)
            ]
            
            bio_content, angles_content, transcript_content = await asyncio.gather(*docs_tasks)
            
            # Summarize content asynchronously
            summarization_tasks = [
                self.summarize_content(bio_content, f"{name}'s Bio v1"),
                self.summarize_content(angles_content, f"{name}'s Angles v1"),
                self.summarize_content(transcript_content, f"{name}'s Transcript")
            ]
            
            content = await asyncio.gather(*summarization_tasks)
            
            # Generate structured data
            structured_data = await self.generate_structured_data(name, content, prompt_content, "v2")
            
            # Create documents and update Airtable
            result = await self.create_documents_and_update(record_id, name, structured_data, "v2")
            
            self.stats['successful'] += 1
            execution_time = time.time() - start_time
            self.stats['execution_times'].append(execution_time)
            
            logger.info(f"Successfully generated bios and angles for {name} (v2) in {execution_time:.2f} seconds")
            
            return {**result, 'execution_time': execution_time}
            
        except Exception as e:
            self.stats['failed'] += 1
            execution_time = time.time() - start_time
            logger.error(f"Error generating Angles and Bio v2 for record {record_id}: {e}")
            return {'status': 'error', 'error': str(e), 'execution_time': execution_time}
        finally:
            self.stats['records_processed'] += 1

    async def filter_and_process_record(self, record_id: str) -> Dict[str, Any]:
        """
        Check if the record has a non-empty 'Transcription with client' field.
        If it does, generate angles and bio (v2). Otherwise, generate angles and bio (v1).
        
        Args:
            record_id: Airtable record ID
            
        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"Filtering record {record_id} by transcription availability...")
            
            loop = asyncio.get_running_loop()
            record = await loop.run_in_executor(
                self.executor,
                lambda: self.airtable_service.get_record_by_id(record_id)
            )
            
            fields = record.get('fields', {})
            
            # Check if 'Angles & Bio Button' is true
            if not fields.get('Angles & Bio Button'):
                logger.info(f"Record {record_id} does not have 'Angles & Bio Button' enabled.")
                return {'status': 'skipped', 'reason': 'Angles & Bio Button not enabled'}
            
            transcription = fields.get('Transcription with client', '')
            has_mock_send = fields.get('Mock Interview Email Send')
            
            if transcription.strip() and has_mock_send:
                # If transcription is not empty and has mock send, generate v2
                return await self.process_record_v2(record_id)
            else:
                # If transcription is empty or no mock send, generate v1
                return await self.process_record_v1(record_id)
                
        except Exception as e:
            logger.error(f"Error in filter_and_process_record for record {record_id}: {e}")
            return {'status': 'error', 'error': str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        avg_time = sum(self.stats['execution_times']) / max(len(self.stats['execution_times']), 1)
        return {
            **self.stats,
            'avg_execution_time': avg_time,
            'timestamp': datetime.now().isoformat()
        }

    def cleanup(self):
        """Clean up resources when done."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


# Legacy function wrappers for backward compatibility
def generate_angles_and_bio_v1(record_id, airtable_service=None):
    """Legacy wrapper for generate_angles_and_bio_v1"""
    processor = AnglesProcessor()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(processor.process_record_v1(record_id))
        return result
    finally:
        processor.cleanup()
        loop.close()

def generate_angles_and_bio_v2(record_id, airtable_service=None):
    """Legacy wrapper for generate_angles_and_bio_v2"""
    processor = AnglesProcessor()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(processor.process_record_v2(record_id))
        return result
    finally:
        processor.cleanup()
        loop.close()

def filter_by_transcription_availability(record_id, airtable_service=None):
    """Legacy wrapper for filter_by_transcription_availability"""
    processor = AnglesProcessor()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(processor.filter_and_process_record(record_id))
        return result
    finally:
        processor.cleanup()
        loop.close()
