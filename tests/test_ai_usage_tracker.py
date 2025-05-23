"""
Test AI Usage Tracker Module

This module provides a simplified version of the AI usage tracker for test scripts.
It separates test usage data from production usage data and provides more accurate token counting.
"""

import os
import csv
import json
import time
import logging
import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import tiktoken  # For OpenAI accurate token counting

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cost rates for different models (same as production)
COST_RATES = {
    # OpenAI models
    'gpt-4o-2024-08-06': {
        'input': 0.0025,  # Per 1K input tokens
        'output': 0.010  # Per 1K output tokens
    },
    # Anthropic models
    'claude-3-5-haiku-20241022': {
        'input': 0.00025,  # Per 1K input tokens
        'output': 0.00125  # Per 1K output tokens
    },
    'claude-3-5-sonnet-20241022': {
        'input': 0.003,  # Per 1K input tokens
        'output': 0.015  # Per 1K output tokens
    },
    # Google models
    'gemini-2.0-flash': {
        'input': 0.00025,  # Per 1K input tokens
        'output': 0.00125  # Per 1K output tokens
    },
    'gemini-1.5-flash': {
        'input': 0.00025,  # Per 1K input tokens
        'output': 0.00125  # Per 1K output tokens
    },

    'o3-mini': {
        'input': 0.0011,  # Per 1K input tokens
        'output': 0.0044  # Per 1K output tokens
    },
    # Default fallback
    'default': {
        'input': 0.01,
        'output': 0.03
    }
}

# Dict mapping model names to their tokenizer settings
MODEL_TOKENIZERS = {
    # OpenAI models use tiktoken
    'gpt-4o-2024-08-06': 'cl100k_base',
    'o3-mini': 'cl100k_base',
    # Others will use custom counting methods
}

class TestAIUsageTracker:
    """A simplified AI usage tracker specifically for test scripts."""
    
    def __init__(self):
        """Initialize the test tracker."""
        # Set up storage in test_scripts directory
        test_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_file = os.path.join(test_dir, 'test_ai_usage_logs.csv')
        self._setup_storage()
        
        # Load tokenizers for OpenAI models
        self.tokenizers = {}
        for model, encoding_name in MODEL_TOKENIZERS.items():
            try:
                self.tokenizers[model] = tiktoken.get_encoding(encoding_name)
            except Exception as e:
                logger.warning(f"Could not load tokenizer for {model}: {e}")
    
    def _setup_storage(self):
        """Set up and initialize the CSV log file."""
        try:
            # Create CSV with headers if it doesn't exist
            if not os.path.exists(self.log_file):
                with open(self.log_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp', 'workflow', 'model', 'tokens_in', 
                        'tokens_out', 'total_tokens', 'cost', 
                        'execution_time_sec', 'test_name', 'record_id'
                    ])
                logger.info(f"Created test log file: {self.log_file}")
        except Exception as e:
            logger.error(f"Failed to set up test log file: {e}")
            raise
    
    def calculate_cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        """Calculate the cost of an API call based on the model and token usage."""
        # Clean model name by removing 'models/' prefix if present
        clean_model = model
        if clean_model.startswith('models/'):
            clean_model = clean_model.replace('models/', '')
        
        # Get rate for the model, or use default if not found
        model_rates = COST_RATES.get(clean_model, COST_RATES['default'])
        
        # Calculate cost (convert from tokens to thousands of tokens)
        input_cost = (tokens_in / 1000) * model_rates['input']
        output_cost = (tokens_out / 1000) * model_rates['output']
        
        return input_cost + output_cost
    
    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens more accurately based on model-specific tokenizers."""
        # Clean model name by removing 'models/' prefix if present
        clean_model = model
        if clean_model.startswith('models/'):
            clean_model = clean_model.replace('models/', '')
            
        # Use tiktoken for OpenAI models
        if clean_model in self.tokenizers:
            return len(self.tokenizers[clean_model].encode(text))
        
        # For Claude models, approximate based on their tokenization approach
        elif 'claude' in clean_model.lower():
            # Claude tends to be ~1.4x GPT-3's token count for English text
            # This is a rough approximation
            import re
            words = len(re.findall(r'\b\w+\b', text))
            return int(words * 1.4)
        
        # For Gemini models, also approximate
        elif 'gemini' in clean_model.lower():
            # Gemini tokenization is roughly comparable to GPT-3.5
            # This is a rough approximation
            import re
            words = len(re.findall(r'\b\w+\b', text))
            return int(words * 1.3)
        
        # Fallback to a better approximation than just dividing by 4
        else:
            # Count words and characters for a better approximation
            import re
            words = len(re.findall(r'\b\w+\b', text))
            chars = len(text)
            
            # Most tokenizers use a mix of word and character level tokens
            # This gives a slightly better estimate than just dividing by 4
            return int((words * 0.6) + (chars * 0.25))
    
    def log_usage(self, 
                workflow: str,
                model: str, 
                tokens_in: int, 
                tokens_out: int, 
                execution_time: float, 
                test_name: str = "default_test",
                record_id: str = None):
        """Log a single AI API usage event to the test CSV file."""
        total_tokens = tokens_in + tokens_out
        cost = self.calculate_cost(model, tokens_in, tokens_out)
        timestamp = datetime.datetime.now().isoformat()
        
        try:
            # Log to CSV file
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    workflow,
                    model,
                    tokens_in,
                    tokens_out,
                    total_tokens,
                    f"{cost:.6f}",
                    f"{execution_time:.3f}",
                    test_name,
                    record_id or "unknown"
                ])
            
            # Also log to console for immediate visibility
            record_info = f" | Record: {record_id}" if record_id else ""
            logger.info(
                f"Test AI Usage: {workflow} | {model} | Tokens: {tokens_in}+{tokens_out}={total_tokens} | "
                f"Cost: ${cost:.6f} | Time: {execution_time:.3f}s | Test: {test_name}{record_info}"
            )
            
            return {
                'timestamp': timestamp,
                'workflow': workflow,
                'model': model,
                'tokens_in': tokens_in,
                'tokens_out': tokens_out,
                'total_tokens': total_tokens,
                'cost': cost,
                'execution_time': execution_time,
                'test_name': test_name,
                'record_id': record_id
            }
        except Exception as e:
            logger.error(f"Error logging test usage: {e}")
            raise
    
    def get_test_results(self, test_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all test results, optionally filtered by test name.
        
        Args:
            test_name: Optional name of the test to filter by
            
        Returns:
            List of test result entries
        """
        if not os.path.exists(self.log_file):
            return []
        
        results = []
        with open(self.log_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter by test name if provided
                if test_name and row['test_name'] != test_name:
                    continue
                
                # Convert numeric fields
                row['tokens_in'] = int(row['tokens_in'])
                row['tokens_out'] = int(row['tokens_out'])
                row['total_tokens'] = int(row['total_tokens'])
                row['cost'] = float(row['cost'])
                row['execution_time_sec'] = float(row['execution_time_sec'])
                
                results.append(row)
        
        return results
    
    def clear_test_results(self, test_name: Optional[str] = None):
        """
        Clear test results, optionally only for a specific test.
        
        Args:
            test_name: Optional name of the test to clear results for
        """
        if not os.path.exists(self.log_file):
            return
        
        if test_name:
            # Read existing data, filter out the test results
            results = []
            with open(self.log_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                results = [row for row in reader if row['test_name'] != test_name]
            
            # Write back the filtered data
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'workflow', 'model', 'tokens_in', 
                    'tokens_out', 'total_tokens', 'cost', 
                    'execution_time_sec', 'test_name', 'record_id'
                ])
                
                for row in results:
                    writer.writerow([
                        row['timestamp'], row['workflow'], row['model'], 
                        row['tokens_in'], row['tokens_out'], row['total_tokens'],
                        row['cost'], row['execution_time_sec'], 
                        row['test_name'], row['record_id']
                    ])
        else:
            # Just recreate the file with headers
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'workflow', 'model', 'tokens_in', 
                    'tokens_out', 'total_tokens', 'cost', 
                    'execution_time_sec', 'test_name', 'record_id'
                ])

# Create a global instance for test scripts
tracker = TestAIUsageTracker() 