"""
Test Script for Pitch Writer (Optimized)

This script tests the pitch_writer_optimized module without modifying Airtable.
Instead, it outputs the results to a CSV file for analysis.

Usage:
    python test_pitch_writer.py --batch-size 2 --concurrency 2 --limit 5 --model gemini_flash --test-name my_test

Options:
    --batch-size: Number of records to process in each batch (default: 1)
    --concurrency: Maximum number of concurrent processes (default: 1)
    --limit: Maximum number of records to process (default: process all)
    --output: Output CSV file path (default: pitch_writer_results_TIMESTAMP.csv)
    --model: LLM model to use (gemini_flash, claude_haiku, o3_mini)
    --test-name: Name to identify this test run
    --debug: Enable debug logging
"""

import os
import sys
import csv
import json
import time
import asyncio
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

# Add parent directory to path so we can import from parent modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import from the pitch_writer_optimized module
from pitch_writer_optimized import (
    PitchWriterProcessor,
    PitchEmail,
    SubjectLine,
    BaseModel,
    Field
)

# Import from the langchain modules
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# Import test-specific AI usage tracker
from test_scripts.test_ai_usage_tracker import tracker as test_tracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"test_pitch_writer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

class TestPitchWriterProcessor(PitchWriterProcessor):
    """
    Modified version of PitchWriterProcessor that doesn't update Airtable
    and outputs results to a CSV file instead.
    """
    
    def __init__(self, model: str = "gemini_flash", test_name: str = "default_test"):
        """
        Initialize with parent class init but allow dynamic LLM selection.
        
        Args:
            model (str): Type of LLM to use. Options: "gemini_flash", "claude_haiku", "o3_mini"
            test_name (str): Name to identify this test run for tracking
        """
        super().__init__()
        self.results = []
        self.model_name = model
        self.test_name = test_name
        self.llm = self.select_llm(model)
        logger.info(f"Test Pitch Writer Processor initialized with {model} model for test '{test_name}'")
        logger.info("Test Pitch Writer Processor initialized in TEST MODE (no Airtable updates)")
    
    def select_llm(self, model: str) -> None:
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
                temperature=0.4
            )
        
        elif model.lower() == "claude_sonnet":
            api_key = os.getenv("ANTHROPIC_API")
            if not api_key:
                raise ValueError("ANTHROPIC_API environment variable not set")
                
            return ChatAnthropic(
                model="claude-3-5-sonnet-20241022", 
                anthropic_api_key=api_key,
                temperature=0.4
            )

        elif model.lower() == "gemini_flash":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
                
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=api_key,
                temperature=0.4
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
            raise ValueError(f"Unsupported model type: {model}. Use 'claude_haiku', 'gemini_flash' or 'o3_mini'")

    async def _run_pitch_email_generation(self, podcast_name: str, host_name: str, episode_title: str,
                                        episode_summary: str, ai_summary: str, client_name: str,
                                        bio: str, bio_summary: str, pitch_topics: str, podcast_id: str) -> tuple:
        """
        Override to use test tracker.
        """
        email_result, token_info, execution_time = await super()._run_pitch_email_generation(
            podcast_name, host_name, episode_title, episode_summary, ai_summary,
            client_name, bio, bio_summary, pitch_topics, podcast_id
        )
        
        # Log to our test-specific tracker instead of the production one
        if email_result:
            model_name = getattr(self.llm, 'model', 'unknown')
            
            # Get actual model name based on provider
            if 'anthropic' in str(self.llm).lower():
                if 'sonnet' in str(self.llm.model).lower():
                    model_name = "claude-3-5-sonnet-20241022"
                else:
                    model_name = "claude-3-5-haiku-20241022"
            elif 'genai' in str(self.llm).lower():
                model_name = "gemini-2.0-flash"
            elif 'openai' in str(self.llm).lower():
                model_name = "o3-mini"
                
            test_tracker.log_usage(
                workflow="pitch_writer",
                model=model_name,
                tokens_in=token_info['input'],
                tokens_out=token_info['output'],
                execution_time=execution_time,
                test_name=self.test_name,
                record_id=podcast_id
            )
        
        return email_result, token_info, execution_time
    
    async def _run_subject_line_generation(self, episode_summary: str, ai_summary: str, podcast_id: str) -> tuple:
        """
        Override to use test tracker.
        """
        subject_result, token_info, execution_time = await super()._run_subject_line_generation(
            episode_summary, ai_summary, podcast_id
        )
        
        # Log to our test-specific tracker instead of the production one
        if subject_result:
            model_name = getattr(self.llm, 'model', 'unknown')
            
            # Get actual model name based on provider
            if 'anthropic' in str(self.llm).lower():
                if 'sonnet' in str(self.llm.model).lower():
                    model_name = "claude-3-5-sonnet-20241022"
                else:
                    model_name = "claude-3-5-haiku-20241022"
            elif 'genai' in str(self.llm).lower():
                model_name = "gemini-2.0-flash"
            elif 'openai' in str(self.llm).lower():
                model_name = "o3-mini"
                
            test_tracker.log_usage(
                workflow="pitch_writer",
                model=model_name,
                tokens_in=token_info['input'],
                tokens_out=token_info['output'],
                execution_time=execution_time,
                test_name=self.test_name,
                record_id=podcast_id
            )
        
        return subject_result, token_info, execution_time

    async def process_single_record(self, cm_record_id: str) -> Dict[str, Any]:
        """
        Process a single Campaign Manager record but don't update Airtable.
        
        Args:
            cm_record_id: The Airtable record ID for the Campaign Manager record
            
        Returns:
            Dict containing the processing results
        """
        result = {
            'record_id': cm_record_id,
            'status': 'Error',
            'podcast_id': None,
            'podcast_name': None,
            'campaign_id': None,
            'campaign_name': None,
            'host_name': None,
            'episode_title': None,
            'subject_line': None,
            'pitch_email': None,
            'error_reason': '',
            'execution_time': 0,
            'email_tokens': 0,
            'subject_tokens': 0,
            'total_tokens': 0,
            'model': self.model_name,  # Add model used
            'test_name': self.test_name,  # Add test name
            'processing_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            start_time = time.time()
            
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
            result['campaign_id'] = campaign_id
            
            campaign_record = self.airtable_service.get_record(self.CAMPAIGNS_TABLE, campaign_id)
            if not campaign_record:
                result['error_reason'] = f"Failed to retrieve Campaign record {campaign_id}"
                return result
            
            campaign_fields = campaign_record.get('fields', {})
            client_names = campaign_fields.get('Name (from Client)', [])
            result['campaign_name'] = client_names[0] if client_names else "Unknown"
            client_name = result['campaign_name']
            
            bio = campaign_fields.get('TextBio', '')
            bio_summary = campaign_fields.get('TextBioSummary', '')
            if not bio_summary:
                bio_summary = bio[:500] + "..." if len(bio) > 500 else bio
            
            # Step 3: Get the Podcast record
            podcast_ids = cm_fields.get('Podcast Name', [])
            if not podcast_ids:
                result['error_reason'] = f"No podcast linked to Campaign Manager record {cm_record_id}"
                return result
            
            podcast_id = podcast_ids[0]
            result['podcast_id'] = podcast_id
            
            podcast_record = self.airtable_service.get_record(self.PODCASTS_TABLE, podcast_id)
            if not podcast_record:
                result['error_reason'] = f"Failed to retrieve Podcast record {podcast_id}"
                return result
            
            podcast_fields = podcast_record.get('fields', {})
            podcast_name = podcast_fields.get('Podcast Name', '')
            result['podcast_name'] = podcast_name
            
            host_name = podcast_fields.get('Host Name', '')
            result['host_name'] = host_name
            
            # Step 4: Get the selected episode details
            pitch_episode_id = cm_fields.get('Pitch Episode')
            if not pitch_episode_id:
                result['error_reason'] = f"No pitch episode selected for this record"
                return result
            
            # Find the episode with the matching Calculation field
            episode_record = None
            episode_ids = podcast_fields.get('Podcast Episodes', [])
            for episode_id in episode_ids:
                episode = self.airtable_service.get_record(self.PODCAST_EPISODES_TABLE, episode_id)
                if episode and episode.get('fields', {}).get('Calculation') == pitch_episode_id:
                    episode_record = episode
                    break
            
            if not episode_record:
                result['error_reason'] = f"Selected episode with ID {pitch_episode_id} not found"
                return result
            
            episode_fields = episode_record.get('fields', {})
            episode_title = episode_fields.get('Episode Title', '')
            result['episode_title'] = episode_title
            episode_summary = episode_fields.get('Summary', '')
            ai_summary = episode_fields.get('AI Summary', '')
            
            # Step 5: Get the pitch topics
            pitch_topics = cm_fields.get('PitchTopics', '')
            if not pitch_topics:
                result['error_reason'] = f"No pitch topics available for this record"
                return result
            
            # Step 6: Generate the email pitch
            email_result, email_tokens, email_time = await self._run_pitch_email_generation(
                podcast_name, host_name, episode_title, episode_summary, ai_summary,
                client_name, bio, bio_summary, pitch_topics, podcast_id
            )
            
            if not email_result:
                result['error_reason'] = "Failed to generate pitch email"
                return result
            
            result['pitch_email'] = email_result
            
            # Step 7: Generate the subject line
            subject_result, subject_tokens, subject_time = await self._run_subject_line_generation(
                episode_summary, ai_summary, podcast_id
            )
            
            if not subject_result:
                result['error_reason'] = "Failed to generate subject line"
                return result
            
            result['subject_line'] = subject_result
            
            # Step 8: SKIPPING Airtable update, just collecting the results
            logger.info(f"[TEST MODE] Would update record {cm_record_id} with email pitch and subject line")
            
            # Update result with success data
            total_tokens = email_tokens['total'] + subject_tokens['total']
            execution_time = time.time() - start_time
            
            result.update({
                'status': 'Success',
                'execution_time': execution_time,
                'email_tokens': email_tokens['total'],
                'subject_tokens': subject_tokens['total'],
                'total_tokens': total_tokens
            })
            
            logger.info(f"Successfully processed record {cm_record_id}")
            
            # Collect the result
            self.results.append(result)
            
            # Add cost calculation from test tracker
            test_results = test_tracker.get_test_results(self.test_name)
            total_cost = 0
            for res in test_results:
                if res.get('record_id') == podcast_id:
                    total_cost += res.get('cost', 0)
            
            result['cost'] = total_cost
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing record {cm_record_id}: {e}", exc_info=True)
            result['error_reason'] = str(e)
            self.results.append(result)
            return result
    
    async def process_all_records(self, max_concurrency=1, batch_size=1, record_limit=None) -> Dict[str, Any]:
        """
        Process Campaign Manager records in the 'Selected Episode' view.
        
        Args:
            max_concurrency: Maximum number of concurrent processes
            batch_size: Number of records per batch
            record_limit: Maximum number of records to process
            
        Returns:
            Statistics about the processing
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'total_email_tokens': 0,
            'total_subject_tokens': 0,
            'total_tokens': 0,
            'total_cost': 0,
            'model': self.model_name,  # Add model information
            'test_name': self.test_name,  # Add test name
            'start_time': time.time(),
            'end_time': None,
            'duration_seconds': 0
        }
        
        try:
            # Clear previous results
            self.results = []
            
            # Fetch records from the 'Selected Episode' view
            logger.info(f"Fetching records from the '{self.PITCH_VIEW}' view")
            records = self.airtable_service.get_records_from_view(
                self.CAMPAIGN_MANAGER_TABLE, self.PITCH_VIEW)
            
            # Apply limit if specified
            if record_limit and record_limit > 0:
                records = records[:record_limit]
                logger.info(f"Limited to {record_limit} records")
            
            logger.info(f"Found {len(records)} record(s) to process")
            
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
                batch_num = i + 1
                logger.info(f"Starting Batch {batch_num}/{len(batches)} ({len(batch)} records)")
                
                if i > 0:
                    # Add a delay between batches
                    logger.info(f"Pausing for 5 seconds before batch {batch_num}...")
                    await asyncio.sleep(5)
                
                start_batch_time = time.time()
                batch_results = await self.process_batch(batch, semaphore)
                batch_duration = time.time() - start_batch_time
                
                logger.info(f"Finished Batch {batch_num}/{len(batches)}. Duration: {batch_duration:.2f}s")
                all_results.extend(batch_results)
                
                # Update stats
                for result in batch_results:
                    stats['total_processed'] += 1
                    
                    if result['status'] == 'Success':
                        stats['successful'] += 1
                        stats['total_email_tokens'] += result['email_tokens']
                        stats['total_subject_tokens'] += result['subject_tokens'] 
                        stats['total_tokens'] += result['total_tokens']
                    else:
                        stats['failed'] += 1
                
                # Pause after every batch
                if batch_num < len(batches):
                    pause_duration = 30
                    logger.info(f"PAUSING for {pause_duration} seconds after processing batch {batch_num}...")
                    await asyncio.sleep(pause_duration)
            
            # Update final stats
            stats['end_time'] = time.time()
            stats['duration_seconds'] = stats['end_time'] - stats['start_time']
            
            # Log statistics
            logger.info("--- Processing Statistics ---")
            logger.info(f"  Total records processed: {stats['total_processed']}")
            logger.info(f"  Successful: {stats['successful']} ({stats['successful']/max(stats['total_processed'], 1)*100:.1f}%)")
            logger.info(f"  Failed: {stats['failed']} ({stats['failed']/max(stats['total_processed'], 1)*100:.1f}%)")
            logger.info(f"  Total email tokens: {stats['total_email_tokens']}")
            logger.info(f"  Total subject line tokens: {stats['total_subject_tokens']}")
            logger.info(f"  Total tokens used: {stats['total_tokens']}")
            logger.info(f"  Total processing duration: {stats['duration_seconds']:.2f} seconds")
            logger.info("-----------------------------")
            
            # Update stats file name with model and test name
            stats_file = f"pitch_writer_stats_{self.model_name}_{self.test_name}_{timestamp}.json"
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
    
    def export_results_to_csv(self, output_file: str) -> None:
        """
        Export the collected results to a CSV file.
        
        Args:
            output_file: Path to the output CSV file
        """
        if not self.results:
            logger.warning("No results to export")
            return
        
        try:
            # Convert results to DataFrame for easier handling
            df = pd.DataFrame(self.results)
            
            # Reorder columns for better readability
            columns_order = [
                'record_id', 'status', 'podcast_id', 'podcast_name',
                'campaign_id', 'campaign_name', 'host_name', 'episode_title',
                'subject_line', 'pitch_email',
                'model', 'test_name', 'cost',  # Add metadata columns
                'email_tokens', 'subject_tokens', 'total_tokens',
                'execution_time', 'processing_timestamp', 'error_reason'
            ]
            
            # Use only columns that exist in the dataframe
            available_columns = [col for col in columns_order if col in df.columns]
            df = df[available_columns]
            
            # Export to CSV
            df.to_csv(output_file, index=False)
            logger.info(f"Results exported to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to export results to CSV: {e}", exc_info=True)


async def run_test(batch_size: int, concurrency: int, limit: Optional[int], output_file: str,
                  model: str = "gemini_flash", test_name: str = "default_test") -> None:
    """
    Run the test processor with the specified parameters.
    
    Args:
        batch_size: Number of records to process in each batch
        concurrency: Maximum number of concurrent processes
        limit: Maximum number of records to process
        output_file: Path to the output CSV file
        model: LLM model to use
        test_name: Name to identify this test run
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
    
    processor = TestPitchWriterProcessor(model=model, test_name=test_name)
    
    start_time = time.time()
    stats = await processor.process_all_records(
        max_concurrency=concurrency,
        batch_size=batch_size,
        record_limit=limit
    )
    end_time = time.time()
    
    total_duration = end_time - start_time
    logger.info(f"Test '{test_name}' completed in {total_duration:.2f} seconds")
    
    # Export results to CSV
    processor.export_results_to_csv(output_file)


def main():
    """Command-line entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test the pitch_writer_optimized module")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of records to process in each batch")
    parser.add_argument("--concurrency", type=int, default=1, help="Maximum number of concurrent processes")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of records to process")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file path")
    parser.add_argument("--model", type=str, default="gemini_flash", 
                        choices=["gemini_flash", "claude_haiku", "o3_mini", "claude_sonnet"], 
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
    
    # Set default output file if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use the exact format test_runner.py expects: {workflow}_{model}_{test_name}_{timestamp}.csv
        args.output = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"pitch_writer_{args.model}_{args.test_name}_{timestamp}.csv"
        )
    
    # Run the test
    logger.info("==============================================")
    logger.info(f"Starting Pitch Writer Test (Model: {args.model}, Test Name: {args.test_name})")
    logger.info("==============================================")
    
    asyncio.run(run_test(
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        limit=args.limit,
        output_file=args.output,
        model=args.model,
        test_name=args.test_name
    ))
    
    logger.info("==============================================")
    logger.info("Test Completed")
    logger.info("==============================================")


if __name__ == "__main__":
    main() 