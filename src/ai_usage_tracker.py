"""
AI Usage Tracker Module

This module provides utilities to track and log AI API usage, including:
- Token usage (input and output)
- Cost calculations
- Time spent on requests
- Model identification
- Endpoint/workflow identification

It creates logs that can be used to analyze usage patterns, costs, and performance.
"""

import os
import csv
import json
import time
import logging
import datetime
import platform
import shutil
from typing import Dict, Any, Optional, List
from pathlib import Path
from google.oauth2.credentials import Credentials
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from google_auth_oauthlib.flow import InstalledAppFlow
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# Constants for cost calculations
# These values can be updated as pricing changes
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
        'input': 0.0001,  # Per 1K input tokens
        'output': 0.0004  # Per 1K output tokens
    },
    'gemini-1.5-flash': {
        'input': 0.000075,  # Per 1K input tokens
        'output': 0.0003  # Per 1K output tokens
    },

    'o3-mini': {
        'input': 0.0011,  # Per 1K input tokens
        'output': 0.0044  # Per 1K output tokens
    },
    # Default fallback
    'default': {
        'input': 0.001,
        'output': 0.003
    }
}


class AIUsageTracker:
    """
    A utility class to track and log AI API usage across the application.
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """Initialize the tracker with storage verification."""
        self._verify_environment()
        self._setup_storage(log_file)
        self._verify_storage_permissions()

        self.GOOGLE_FOLDER_ID = os.getenv('PGL_AI_DRIVE_FOLDER_ID')
        self.BACKUP_INTERVAL = 3600  # 1 hour in seconds
        self.last_backup_time = None

        self._init_google_drive()

    def _verify_environment(self):
        """Verify and log details about the running environment."""
        env_info = {
            'Platform': platform.platform(),
            'Python Version': platform.python_version(),
            'REPL_HOME': os.getenv('REPL_HOME', 'Not running on Replit'),
            'REPL_ID': os.getenv('REPL_ID', 'Not available'),
            'REPL_SLUG': os.getenv('REPL_SLUG', 'Not available'),
            'Working Directory': os.getcwd(),
        }
        
        logger.info("Environment Information:")
        for key, value in env_info.items():
            logger.info(f"  {key}: {value}")

    def _setup_storage(self, log_file: Optional[str]):
        """Set up and verify storage locations."""
        self.replit_home = os.getenv('REPL_HOME', '')
        
        if self.replit_home:
            # We're running on Replit
            logger.info(f"Running on Replit (REPL_HOME: {self.replit_home})")
            
            # Set up persistent directory
            persistent_dir = os.path.join(self.replit_home, '.persistent')
            os.makedirs(persistent_dir, exist_ok=True)
            
            # Verify persistent directory
            if not os.path.exists(persistent_dir):
                raise RuntimeError(f"Failed to create persistent directory: {persistent_dir}")
            
            # Set up log file path
            self.log_file = os.path.join(persistent_dir, 'ai_usage_logs.csv')
            logger.info(f"Using Replit persistent storage at: {self.log_file}")
            
            # Create a test file to verify write permissions
            test_file = os.path.join(persistent_dir, 'storage_test.txt')
            try:
                with open(test_file, 'w') as f:
                    f.write('Storage test')
                os.remove(test_file)
                logger.info("Successfully verified write permissions in persistent directory")
            except Exception as e:
                logger.error(f"Failed to write to persistent directory: {e}")
                raise
        else:
            # Local development
            self.log_file = log_file or 'ai_usage_logs.csv'
            logger.info(f"Running locally, using storage at: {self.log_file}")
        
        # Ensure the directory exists
        log_dir = os.path.dirname(os.path.abspath(self.log_file))
        os.makedirs(log_dir, exist_ok=True)
        
        # Verify the directory was created
        if not os.path.exists(log_dir):
            raise RuntimeError(f"Failed to create log directory: {log_dir}")
        
        logger.info(f"Storage directory verified: {log_dir}")
        
        # Set up backup path
        self.backup_file = os.path.join(
            os.path.dirname(self.log_file),
            'ai_usage_logs_backup.csv'
        )

    def _verify_storage_permissions(self):
        """Verify read/write permissions and CSV file integrity."""
        try:
            # Check if file exists
            file_exists = os.path.exists(self.log_file)
            logger.info(f"Log file exists: {file_exists}")
            
            if file_exists:
                # Verify file is readable
                with open(self.log_file, 'r') as f:
                    header = f.readline().strip()
                logger.info("Successfully read from existing log file")
                
                # Verify CSV structure
                with open(self.log_file, 'r', newline='') as f:
                    reader = csv.reader(f)
                    header_row = next(reader, None)
                    if header_row != [
                        'timestamp', 'workflow', 'model', 'tokens_in', 
                        'tokens_out', 'total_tokens', 'cost', 
                        'execution_time_sec', 'endpoint', 'podcast_id'
                    ]:
                        logger.warning("CSV header structure doesn't match expected format")
                        # Create backup and new file
                        self._create_backup_and_new_file()
                    else:
                        logger.info("CSV structure verified")
            else:
                # Create new file with headers
                self._create_new_file()
            
            # Verify write permissions
            self._verify_write_permissions()
            
        except Exception as e:
            logger.error(f"Storage verification failed: {e}")
            raise

    def _create_new_file(self):
        """Create a new log file with headers."""
        try:
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'workflow', 'model', 'tokens_in', 
                    'tokens_out', 'total_tokens', 'cost', 
                    'execution_time_sec', 'endpoint', 'podcast_id'
                ])
            logger.info(f"Created new log file: {self.log_file}")
        except Exception as e:
            logger.error(f"Failed to create new log file: {e}")
            raise

    def _create_backup_and_new_file(self):
        """Create a backup of existing file and create new one."""
        try:
            if os.path.exists(self.log_file):
                # Create backup with timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{self.backup_file}_{timestamp}"
                shutil.copy2(self.log_file, backup_path)
                logger.info(f"Created backup at: {backup_path}")
            
            # Create new file
            self._create_new_file()
            
        except Exception as e:
            logger.error(f"Failed to create backup and new file: {e}")
            raise

    def _verify_write_permissions(self):
        """Verify write permissions by attempting to write and read a test entry."""
        try:
            test_entry = {
                'timestamp': datetime.datetime.now().isoformat(),
                'workflow': 'storage_test',
                'model': 'test_model',
                'tokens_in': 0,
                'tokens_out': 0,
                'total_tokens': 0,
                'cost': 0.0,
                'execution_time_sec': 0.0,
                'endpoint': 'test',
                'podcast_id': 'test'
            }
            
            # Write test entry
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(list(test_entry.values()))
            
            # Verify entry was written
            with open(self.log_file, 'r', newline='') as f:
                reader = csv.reader(f)
                rows = list(reader)
                last_row = rows[-1]
                if last_row[1] == 'storage_test':
                    logger.info("Write permissions verified successfully")
                    
                    # Remove test entry
                    with open(self.log_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows(rows[:-1])
                else:
                    raise RuntimeError("Failed to verify written test entry")
                    
        except Exception as e:
            logger.error(f"Write permission verification failed: {e}")
            raise

    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about the current storage setup."""
        try:
            file_size = os.path.getsize(self.log_file) if os.path.exists(self.log_file) else 0
            
            # Count entries
            num_entries = 0
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r', newline='') as f:
                    num_entries = sum(1 for _ in csv.reader(f)) - 1  # Subtract header
            
            storage_info = {
                'environment': 'replit' if self.replit_home else 'local',
                'log_file_path': self.log_file,
                'file_exists': os.path.exists(self.log_file),
                'file_size_bytes': file_size,
                'num_entries': num_entries,
                'is_writable': os.access(os.path.dirname(self.log_file), os.W_OK),
                'is_readable': os.access(self.log_file, os.R_OK) if os.path.exists(self.log_file) else False,
                'backup_file_path': self.backup_file,
                'last_backup_time': self.last_backup_time,
                'google_drive_connected': self.drive_service is not None
            }
            
            return storage_info
            
        except Exception as e:
            logger.error(f"Failed to get storage info: {e}")
            return {'error': str(e)}

    def _init_google_drive(self):
        """Initialize connection to Google Drive."""
        try:
            service_account_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'service-account-key.json')
            if os.path.exists(service_account_path):
                creds = ServiceAccountCredentials.from_service_account_file(
                    service_account_path,
                    scopes=['https://www.googleapis.com/auth/drive.file']
                )
                self.drive_service = build('drive', 'v3', credentials=creds)
                logger.info(f"Connected to Google Drive using service account from: {service_account_path}")
                return
            else:
                logger.warning(f"Service account file not found at: {service_account_path}")

            # If no service account, try OAuth
            creds = None
            if os.path.exists('token.json'):
                creds = Credentials.from_authorized_user_file('token.json', 
                    ['https://www.googleapis.com/auth/drive.file'])

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        'credentials.json',
                        ['https://www.googleapis.com/auth/drive.file']
                    )
                    creds = flow.run_local_server(port=0)
                
                with open('token.json', 'w') as token:
                    token.write(creds.to_json())

            self.drive_service = build('drive', 'v3', credentials=creds)
            logger.info("Connected to Google Drive using OAuth")

        except Exception as e:
            logger.error(f"Failed to initialize Google Drive connection: {e}")
            self.drive_service = None

    def _backup_to_drive(self):
        """Backup the CSV file to Google Drive if enough time has passed."""
        if not self.drive_service or not self.GOOGLE_FOLDER_ID:
            logger.warning("Skipping backup: Drive service or folder ID not available")
            return

        current_time = time.time()
        if (self.last_backup_time and 
            current_time - self.last_backup_time < self.BACKUP_INTERVAL):
            return

        try:
            # Check if file exists and has content before backing up
            if not os.path.exists(self.log_file) or os.path.getsize(self.log_file) == 0:
                logger.warning("Skipping backup: Log file is empty or doesn't exist")
                return

            # Prepare the file metadata with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_metadata = {
                'name': f'ai_usage_logs_{timestamp}.csv',
                'parents': [self.GOOGLE_FOLDER_ID],
                'description': f'AI Usage Logs backup from {timestamp}'
            }

            # Create a MediaFileUpload object
            media = MediaFileUpload(
                self.log_file,
                mimetype='text/csv',
                resumable=True
            )

            # Upload the file
            file = self.drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id,name'
            ).execute()

            logger.info(f"Backed up AI usage logs to Google Drive: {file.get('name')} (ID: {file.get('id')})")
            self.last_backup_time = current_time

        except Exception as e:
            logger.error(f"Failed to backup to Google Drive: {e}")

    def calculate_cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        """
        Calculate the cost of an API call based on the model and token usage.
        
        Args:
            model: The AI model used (e.g., 'gpt-4o-2024-08-06')
            tokens_in: Number of input tokens
            tokens_out: Number of output tokens
            
        Returns:
            float: Estimated cost in USD
        """
        # Get rate for the model, or use default if not found
        model_rates = COST_RATES.get(model, COST_RATES['default'])
        
        # Calculate cost (convert from tokens to thousands of tokens)
        input_cost = (tokens_in / 1000) * model_rates['input']
        output_cost = (tokens_out / 1000) * model_rates['output']
        
        return input_cost + output_cost
    
    def log_usage(self, 
                workflow: str,
                model: str, 
                tokens_in: int, 
                tokens_out: int, 
                execution_time: float, 
                endpoint: str = "unknown",
                podcast_id: str = None):
        """Log a single AI API usage event to the CSV file."""
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
                    endpoint,
                    podcast_id or "unknown"
                ])
            
            # Attempt to backup to Google Drive
            self._backup_to_drive()
            
            # Also log to console for immediate visibility
            record_info = f" | Record: {podcast_id}" if podcast_id else ""
            logger.info(
                f"AI Usage: {workflow} | {model} | Tokens: {tokens_in}+{tokens_out}={total_tokens} | "
                f"Cost: ${cost:.6f} | Time: {execution_time:.3f}s{record_info}"
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
                'endpoint': endpoint,
                'podcast_id': podcast_id
            }
        except Exception as e:
            logger.error(f"Error logging usage: {e}")
            # If primary location fails, try backup location
            if self.log_file != 'ai_usage_logs_backup.csv':
                self.log_file = 'ai_usage_logs_backup.csv'
                logger.info("Switching to backup log file")
                return self.log_usage(workflow, model, tokens_in, tokens_out, 
                                   execution_time, endpoint, podcast_id)
            raise
    
    def generate_report(self, 
                       start_date: Optional[str] = None, 
                       end_date: Optional[str] = None,
                       group_by: str = 'model') -> Dict[str, Any]:
        """
        Generate a summary report of AI usage within a date range.
        
        Args:
            start_date: ISO format date string (YYYY-MM-DD)
            end_date: ISO format date string (YYYY-MM-DD)
            group_by: Field to group data by ('model', 'workflow', 'endpoint', or 'podcast_id')
            
        Returns:
            Dict containing summary statistics
        """
        if not os.path.exists(self.log_file):
            return {"error": "No log file exists yet"}
        
        # Convert date strings to datetime objects if provided
        start_datetime = None
        if start_date:
            start_datetime = datetime.datetime.fromisoformat(start_date)
        
        end_datetime = None
        if end_date:
            end_datetime = datetime.datetime.fromisoformat(end_date)
        
        # Read and filter log entries
        entries = []
        with open(self.log_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert timestamp to datetime for filtering
                entry_timestamp = datetime.datetime.fromisoformat(row['timestamp'])
                
                # Apply date filters
                if start_datetime and entry_timestamp < start_datetime:
                    continue
                if end_datetime and entry_timestamp > end_datetime:
                    continue
                
                # Convert numeric fields
                row['tokens_in'] = int(row['tokens_in'])
                row['tokens_out'] = int(row['tokens_out'])
                row['total_tokens'] = int(row['total_tokens'])
                row['cost'] = float(row['cost'])
                row['execution_time_sec'] = float(row['execution_time_sec'])
                
                entries.append(row)
        
        # If no entries found after filtering
        if not entries:
            return {
                "start_date": start_date,
                "end_date": end_date,
                "total_entries": 0,
                "total_cost": 0,
                "summary": "No data found for the specified period"
            }
        
        # Calculate overall statistics
        total_cost = sum(entry['cost'] for entry in entries)
        total_tokens = sum(entry['total_tokens'] for entry in entries)
        total_calls = len(entries)
        
        # Group data based on the specified field
        groups = {}
        for entry in entries:
            group_key = entry[group_by]
            if group_key not in groups:
                groups[group_key] = {
                    'calls': 0,
                    'tokens_in': 0,
                    'tokens_out': 0,
                    'total_tokens': 0,
                    'cost': 0,
                    'avg_time': 0
                }
            
            groups[group_key]['calls'] += 1
            groups[group_key]['tokens_in'] += entry['tokens_in']
            groups[group_key]['tokens_out'] += entry['tokens_out']
            groups[group_key]['total_tokens'] += entry['total_tokens']
            groups[group_key]['cost'] += entry['cost']
            groups[group_key]['avg_time'] += entry['execution_time_sec']
        
        # Calculate averages for each group
        for group in groups.values():
            group['avg_time'] = group['avg_time'] / group['calls']
        
        # Prepare the final report
        report = {
            "start_date": start_date,
            "end_date": end_date,
            "total_entries": total_calls,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "grouped_by": group_by,
            "groups": groups
        }
        
        return report
    
    def get_record_cost_report(self, podcast_id: str) -> Dict[str, Any]:
        """
        Generate a detailed cost report for a specific Airtable record ID.
        
        Args:
            podcast_id: The Airtable podcast ID to generate a report for
            
        Returns:
            Dict containing cost and usage information for this record
        """
        if not os.path.exists(self.log_file):
            return {"error": "No log file exists yet"}
        
        # Read all entries from the log file
        all_entries = []
        with open(self.log_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                row['tokens_in'] = int(row['tokens_in'])
                row['tokens_out'] = int(row['tokens_out'])
                row['total_tokens'] = int(row['total_tokens'])
                row['cost'] = float(row['cost'])
                row['execution_time_sec'] = float(row['execution_time_sec'])
                all_entries.append(row)
        
        # Filter entries for the specified record ID
        record_entries = [entry for entry in all_entries if entry.get('podcast_id') == podcast_id]
        
        if not record_entries:
            return {
                "podcast_id": podcast_id,
                "status": "not_found",
                "message": f"No usage data found for record ID {podcast_id}"
            }
        
        # Calculate totals
        total_cost = sum(entry['cost'] for entry in record_entries)
        total_tokens_in = sum(entry['tokens_in'] for entry in record_entries)
        total_tokens_out = sum(entry['tokens_out'] for entry in record_entries)
        total_tokens = sum(entry['total_tokens'] for entry in record_entries)
        total_calls = len(record_entries)
        
        # Group by workflow stage to see the different processes
        stages = {}
        for entry in record_entries:
            workflow = entry['workflow']
            if workflow not in stages:
                stages[workflow] = {
                    'calls': 0,
                    'tokens_in': 0,
                    'tokens_out': 0,
                    'total_tokens': 0,
                    'cost': 0.0  # Ensure cost is initialized as float
                }
            
            stages[workflow]['calls'] += 1
            stages[workflow]['tokens_in'] += entry['tokens_in']
            stages[workflow]['tokens_out'] += entry['tokens_out']
            stages[workflow]['total_tokens'] += entry['total_tokens']
            stages[workflow]['cost'] += float(entry['cost'] or 0.0)  # Ensure cost is a float
        
        # Create timeline of operations
        timeline = []
        for entry in sorted(record_entries, key=lambda x: x['timestamp']):
            timeline.append({
                'timestamp': entry['timestamp'],
                'workflow': entry['workflow'],
                'model': entry['model'],
                'tokens_in': entry['tokens_in'] or 0,
                'tokens_out': entry['tokens_out'] or 0,
                'total_tokens': entry['total_tokens'] or 0,
                'cost': float(entry['cost'] or 0.0)  # Ensure cost is a float
            })
        
        # Prepare the report
        report = {
            "podcast_id": podcast_id,
            "total_cost": float(total_cost or 0.0),  # Ensure total_cost is a float
            "total_tokens_in": total_tokens_in or 0,
            "total_tokens_out": total_tokens_out or 0,
            "total_tokens": total_tokens or 0,
            "total_calls": total_calls or 0,
            "workflow_stages": stages,
            "timeline": timeline
        }
        
        return report

# Create a global instance that can be imported throughout the application
tracker = AIUsageTracker() 