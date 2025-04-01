"""
Test Runner for LLM Models

This module provides FastAPI endpoints for running tests on different LLM models
for various workflows. It handles test execution, tracking, and result downloads.
"""

import os
import sys
import csv
import json
import uuid
import logging
import asyncio
import threading
from fastapi import APIRouter, Request, Depends, HTTPException, Response
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import pandas as pd

# Add parent directory to path so we can import from parent modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import auth middleware for proper authentication
from auth_middleware import get_admin_user

# Import the test-specific AI usage tracker
from test_scripts.test_ai_usage_tracker import tracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dictionary to store active tests
active_tests = {}

# Dictionary mapping workflow names to their test scripts
WORKFLOW_MODULES = {
    'determine_fit': 'test_determine_fit',
    'summary_host_guest': 'test_summary_guest_identification',
    'pitch_episode_angle': 'test_pitch_episode_selection',
    'pitch_writer': 'test_pitch_writer'
}

# Models available for testing
AVAILABLE_MODELS = {
    'gemini_flash': "Gemini 2.0 Flash",
    'claude_haiku': "Claude 3.5 Haiku",
    'claude_sonnet': "Claude 3.5 Sonnet",
    'o3_mini': "OpenAI o3-mini"
}

# Create an API router
router = APIRouter(prefix="/api", tags=["test"])

# Pydantic model for test run request
class TestRunRequest(BaseModel):
    workflow: str
    model: str
    test_name: str
    limit: Optional[int] = 10  # Default to 10 records to avoid processing too many
    batch_size: Optional[int] = 5
    concurrency: Optional[int] = 3

# Pydantic model for test status
class TestStatus(BaseModel):
    test_id: str
    workflow: str
    model: str
    status: str
    progress: Optional[float] = None
    test_name: str

# Functions to run and manage tests
def run_test_thread(test_id, workflow, model, test_name, limit=None, batch_size=5, concurrency=3):
    """Thread function to run a test workflow with the specified model."""
    try:
        # Update test status to running
        active_tests[test_id]['status'] = 'Running'
        
        # Import the appropriate test module
        module_name = WORKFLOW_MODULES.get(workflow)
        if not module_name:
            raise ValueError(f"Unknown workflow: {workflow}")
        
        # Dynamically import the module
        module = __import__(f"test_scripts.{module_name}", fromlist=['run_test'])
        
        # Prepare arguments for the test
        kwargs = {
            'model': model,
            'batch_size': batch_size,
            'concurrency': concurrency,
            'test_name': test_name,
            'limit': limit if limit is not None else None  # Explicitly pass limit even if None
        }
        
        # Set up output CSV file with test name included
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            f"{workflow}_{model}_{test_name}_{timestamp}.csv"
        )
        kwargs['output_file'] = output_file
        
        # Store the output file path for downloading later
        active_tests[test_id]['output_file'] = output_file
        
        # Run the test using asyncio run
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Add signal handlers to properly shutdown the loop
        import signal
        
        # Define a handler for graceful shutdown
        def handle_shutdown(signame):
            logger.info(f"Received signal {signame} - shutting down test gracefully")
            active_tests[test_id]['status'] = 'Stopping'
            # Set some event, flag or internal state to signal the test to stop
            # We need to ensure tasks complete or are cancelled properly
            for task in asyncio.all_tasks(loop):
                task.cancel()
        
        # Add signal handlers for graceful shutdown
        for signame in ('SIGINT', 'SIGTERM'):
            try:
                loop.add_signal_handler(
                    getattr(signal, signame),
                    lambda signame=signame: handle_shutdown(signame)
                )
            except (NotImplementedError, AttributeError):
                # Windows doesn't support SIGTERM
                logger.debug(f"Could not add handler for {signame}")
        
        try:
            # Run the test function
            if hasattr(module, 'run_test'):
                logger.info(f"Running test with kwargs: {kwargs}")
                loop.run_until_complete(module.run_test(**kwargs))
            else:
                # This fallback should no longer be needed
                logger.warning(f"Module {module_name} has no run_test function, using fallback")
                from test_scripts.test_determine_fit import run_test
                loop.run_until_complete(run_test(**kwargs))
            
            # Mark test as completed
            active_tests[test_id]['status'] = 'Completed'
            logger.info(f"Test {test_id} ({test_name}) completed successfully")
            
            # Process results if the output file exists
            if os.path.exists(output_file):
                process_test_results(test_id, test_name, output_file, model, workflow)
            else:
                logger.warning(f"Test {test_id} ({test_name}) completed but no output file was found")
        
        except asyncio.CancelledError:
            logger.info(f"Test {test_id} was cancelled")
            active_tests[test_id]['status'] = 'Cancelled'
        
        finally:
            # Cancel all running tasks
            for task in asyncio.all_tasks(loop):
                task.cancel()
            
            # Run until all tasks are cancelled
            try:
                loop.run_until_complete(asyncio.gather(*asyncio.all_tasks(loop), return_exceptions=True))
            except Exception as e:
                logger.error(f"Error during task cancellation: {e}")
            
            # Close the loop
            loop.close()
            
    except Exception as e:
        logger.error(f"Error running test {test_id} ({test_name}): {e}", exc_info=True)
        active_tests[test_id]['status'] = 'Error'
        active_tests[test_id]['error'] = str(e)
    
    # After some time, clean up completed or error tests
    def cleanup_test():
        import time
        time.sleep(300)  # Keep completed tests for 5 minutes
        if test_id in active_tests:
            del active_tests[test_id]
    
    # Make the cleanup thread a daemon so it doesn't prevent shutdown
    cleanup_thread = threading.Thread(target=cleanup_test, daemon=True)
    cleanup_thread.start()

def process_test_results(test_id, test_name, output_file, model, workflow):
    """Process the test results and store summary information."""
    try:
        # Read the CSV file
        df = pd.read_csv(output_file)
        
        # Add model and test name columns if they don't exist
        if 'model' not in df.columns:
            df['model'] = model
        if 'test_name' not in df.columns:
            df['test_name'] = test_name
        
        # Save enhanced CSV
        df.to_csv(output_file, index=False)
        
        # Store summary information in active_tests
        active_tests[test_id]['records'] = len(df)
        
        # Get detailed usage metrics from the test tracker
        test_results = tracker.get_test_results(test_name)
        if test_results:
            total_tokens = sum(result['total_tokens'] for result in test_results)
            total_cost = sum(result['cost'] for result in test_results)
            
            active_tests[test_id]['total_tokens'] = total_tokens
            active_tests[test_id]['cost'] = total_cost
        else:
            # Fallback to basic metrics if test tracker results aren't available
            active_tests[test_id]['total_tokens'] = df.get('total_tokens', 0).sum() if 'total_tokens' in df.columns else 0
            active_tests[test_id]['cost'] = df.get('cost', 0).sum() if 'cost' in df.columns else 0
    
    except Exception as e:
        logger.error(f"Error processing test results for {test_id}: {e}", exc_info=True)
        active_tests[test_id]['error'] = f"Error processing results: {str(e)}"

# API endpoints
@router.post("/run_test")
async def start_test(request: TestRunRequest, user: dict = Depends(get_admin_user)):
    """
    Start a test run for a specific workflow and model.
    Admin access required.
    """
    try:
        # Validate workflow
        if request.workflow not in WORKFLOW_MODULES:
            return JSONResponse(
                content={"error": f"Unknown workflow: {request.workflow}. Available workflows: {list(WORKFLOW_MODULES.keys())}"},
                status_code=400
            )
        
        # Validate model
        if request.model not in AVAILABLE_MODELS:
            return JSONResponse(
                content={"error": f"Unknown model: {request.model}. Available models: {list(AVAILABLE_MODELS.keys())}"},
                status_code=400
            )
        
        # Generate a unique test ID
        test_id = str(uuid.uuid4())
        
        # Store test information
        active_tests[test_id] = {
            'test_id': test_id,
            'workflow': request.workflow,
            'model': request.model,
            'status': 'Starting',
            'test_name': request.test_name,
            'started_at': datetime.now().isoformat(),
            'started_by': user['username']
        }
        
        # Start the test in a separate thread
        thread = threading.Thread(
            target=run_test_thread,
            args=(
                test_id, 
                request.workflow, 
                request.model, 
                request.test_name,
                request.limit,
                request.batch_size,
                request.concurrency
            )
        )
        thread.start()
        
        logger.info(f"Started test {test_id} for workflow {request.workflow} with model {request.model}")
        
        return {
            "test_id": test_id,
            "message": f"Test started for workflow {request.workflow} with model {request.model}",
            "status": "starting"
        }
        
    except Exception as e:
        logger.error(f"Error starting test: {e}", exc_info=True)
        return JSONResponse(
            content={"error": f"Failed to start test: {str(e)}"},
            status_code=500
        )

@router.post("/stop_test/{test_id}")
async def stop_test(test_id: str, user: dict = Depends(get_admin_user)):
    """
    Stop a running test.
    Admin access required.
    """
    try:
        if test_id not in active_tests:
            return JSONResponse(
                content={"error": f"Test {test_id} not found"},
                status_code=404
            )
        
        # Update test status to stopping
        active_tests[test_id]['status'] = 'Stopping'
        
        # Actually stopping the test would require a more complex mechanism
        # For now, we just mark it as stopping and let it finish
        
        logger.info(f"Test {test_id} marked as stopping by {user['username']}")
        
        return {
            "message": f"Test {test_id} is being stopped",
            "status": "stopping"
        }
        
    except Exception as e:
        logger.error(f"Error stopping test {test_id}: {e}", exc_info=True)
        return JSONResponse(
            content={"error": f"Failed to stop test: {str(e)}"},
            status_code=500
        )

@router.get("/active_tests")
async def get_active_tests(user: dict = Depends(get_admin_user)):
    """
    Get a list of all active tests.
    Admin access required.
    """
    try:
        result = []
        for test_id, test_info in active_tests.items():
            # Create a simplified representation for the response
            result.append({
                'test_id': test_id,
                'workflow': test_info.get('workflow'),
                'model': test_info.get('model'),
                'status': test_info.get('status'),
                'progress': test_info.get('progress'),
                'test_name': test_info.get('test_name')
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving active tests: {e}", exc_info=True)
        return JSONResponse(
            content={"error": f"Failed to retrieve active tests: {str(e)}"},
            status_code=500
        )

@router.get("/completed_tests")
async def get_completed_tests(user: dict = Depends(get_admin_user)):
    """
    Get a list of completed tests.
    Admin access required.
    """
    try:
        result = []
        for test_id, test_info in active_tests.items():
            # Only include completed tests that have output files
            if test_info.get('status') == 'Completed' and 'output_file' in test_info:
                result.append({
                    'test_id': test_id,
                    'test_name': test_info.get('test_name'),
                    'workflow': test_info.get('workflow'),
                    'model': test_info.get('model'),
                    'records': test_info.get('records', 0),
                    'total_tokens': test_info.get('total_tokens', 0),
                    'cost': test_info.get('cost', 0),
                    'output_file': test_info.get('output_file')
                })
        
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving completed tests: {e}", exc_info=True)
        return JSONResponse(
            content={"error": f"Failed to retrieve completed tests: {str(e)}"},
            status_code=500
        )

@router.get("/download_results/{test_name}")
async def download_results(test_name: str, user: dict = Depends(get_admin_user)):
    """
    Download test results as a CSV file.
    Admin access required.
    """
    try:
        # Find the test by name
        found_test = None
        for test_info in active_tests.values():
            if test_info.get('test_name') == test_name and 'output_file' in test_info:
                found_test = test_info
                break
        
        if not found_test:
            return JSONResponse(
                content={"error": f"Test results for '{test_name}' not found"},
                status_code=404
            )
        
        output_file = found_test['output_file']
        
        if not os.path.exists(output_file):
            return JSONResponse(
                content={"error": f"Result file not found on server"},
                status_code=404
            )
        
        # Create a filename for download
        workflow = found_test.get('workflow', 'unknown')
        model = found_test.get('model', 'unknown')
        download_filename = f"{workflow}_{model}_{test_name}_results.csv"
        
        return FileResponse(
            path=output_file,
            filename=download_filename,
            media_type='text/csv'
        )
        
    except Exception as e:
        logger.error(f"Error downloading results for {test_name}: {e}", exc_info=True)
        return JSONResponse(
            content={"error": f"Failed to download results: {str(e)}"},
            status_code=500
        )

def register_routes(app):
    """Register the test runner routes with the main FastAPI app."""
    app.include_router(router)
    
    # Add the test dashboard page
    @app.get("/llm-test", tags=["test"])
    async def llm_test_dashboard(request: Request, user: dict = Depends(get_admin_user)):
        from fastapi.templating import Jinja2Templates
        templates = Jinja2Templates(directory="templates")
        return templates.TemplateResponse("llm_test_dashboard.html", {
            "request": request,
            "username": user["username"],
            "role": user["role"]
        })
    
    logger.info("Test Runner routes registered with the FastAPI app") 