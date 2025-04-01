"""
Unified FastAPI Application

Author: Paschal Okonkwor
Date: 2025-01-06
"""

import os
import logging
import uuid
from fastapi import FastAPI, Request, Query, Response, status, Form, Depends
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional
from datetime import datetime

# Import the authentication middleware
from auth_middleware import (
    AuthMiddleware, 
    authenticate_user, 
    create_session, 
    get_current_user,
    get_admin_user
)

# Import the task manager
from task_manager import task_manager

# Import the functions that handle specific tasks
from webhook_handler import poll_airtable_and_process, poll_podcast_search_database, enrich_host_name
from summary_guest_identification_optimized import PodcastProcessor  # Import the optimized version
from determine_fit_optimized import determine_fit  # Import the optimized version
from pitch_episode_selection_optimized import pitch_episode_selection  # Import the optimized version
from pitch_writer_optimized import pitch_writer  # Import the optimized version
from send_pitch_to_instantly import send_pitch_to_instantly
from instantly_email_sent import update_airtable_when_email_sent
from instantly_response import update_correspondent_on_airtable
from fetch_episodes import get_podcast_episodes
from podcast_note_transcriber import get_podcast_audio_transcription, transcribe_endpoint

# Import the AI usage tracker
from ai_usage_tracker import tracker as ai_tracker

# Import the test runner (for LLM model testing)
from test_scripts.test_runner import register_routes as register_test_routes

# -------------------------------------------------------------------
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# -------------------------------------------------------------------

app = FastAPI()

# Add authentication middleware
app.add_middleware(AuthMiddleware)

# Initialize Jinja2Templates for HTML rendering
templates = Jinja2Templates(directory="templates")

# Register custom filters for templates
def format_number(value):
    """Format a number with comma separators"""
    return f"{value:,}"

def format_datetime(timestamp):
    """Convert ISO timestamp to readable format"""
    dt = datetime.fromisoformat(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

templates.env.filters["format_number"] = format_number
templates.env.filters["format_datetime"] = format_datetime

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/login")
def login_page(request: Request):
    """Render the login page"""
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...)
):
    """Handle login form submission"""
    # Authenticate user
    role = authenticate_user(username, password)
    
    if not role:
        # Authentication failed
        return templates.TemplateResponse(
            "login.html", 
            {
                "request": request, 
                "error": "Invalid username or password"
            }
        )
    
    # Create session
    session_id = create_session(username, role)
    
    # Create response with redirect
    response = RedirectResponse(url="/", status_code=303)
    
    # Set session cookie
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        secure=True,  # Set to False for http development
        samesite="lax",
        max_age=3600  # 1 hour
    )
    
    return response


@app.get("/logout")
def logout(request: Request):
    """Log out the user by clearing the session cookie"""
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(key="session_id")
    return response


@app.get("/")
def index(request: Request, user: dict = Depends(get_current_user)):
    """
    Root endpoint that renders the main dashboard HTML.
    Requires authentication.
    """
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request,
            "username": user["username"],
            "role": user["role"]
        }
    )


@app.get("/api-status")
def api_status():
    """
    API status endpoint that returns a simple JSON message.
    This endpoint is public.
    """
    return {"message": "PGL Automation API is running (FastAPI version)!"}


@app.get("/trigger_automation")
def trigger_automation(
        action: str = Query(...,
                            description="Name of the automation to trigger"),
        id: Optional[str] = Query(
            None, description="Record ID for the automation if needed")):
    """
    A single endpoint to trigger one of multiple automation functions.
    Returns a task ID that can be used to stop the automation if needed.
    This endpoint is publicly accessible (no auth required).
    """
    try:
        # Generate a unique task ID
        task_id = str(uuid.uuid4())
        
        # Register the task before starting
        task_manager.start_task(task_id, action)
        
        # Start the task in a separate thread
        import threading
        
        def run_task():
            try:
                # Get the stop flag for this task
                stop_flag = task_manager.get_stop_flag(task_id)
                if not stop_flag:
                    logger.error(f"Could not get stop flag for task {task_id}")
                    return
                
                if action == 'generate_bio_angles':
                    poll_airtable_and_process(stop_flag)
                
                elif action == 'mipr_podcast_search':
                    if not id:
                        raise ValueError("Missing 'id' parameter for MIPR Podcast Search automation!")
                    poll_podcast_search_database(id, stop_flag)
                
                elif action == 'fetch_podcast_episodes':
                    get_podcast_episodes(stop_flag)
                
                elif action == 'summary_host_guest':
                    run_summary_host_guest_optimized(stop_flag)
                
                elif action == 'determine_fit':
                    determine_fit(stop_flag)
                
                elif action == 'pitch_episode_angle':
                    pitch_episode_selection(stop_flag)
                
                elif action == 'pitch_writer':
                    pitch_writer(stop_flag)
                
                elif action == 'send_pitch':
                    send_pitch_to_instantly(stop_flag)
                
                elif action == 'enrich_host_name':
                    enrich_host_name(stop_flag)
                
                elif action == 'transcribe_podcast':
                    run_transcription_task(stop_flag)
                
                else:
                    raise ValueError(f"Invalid action: {action}")
                
            except Exception as e:
                logger.error(f"Error in task {task_id}: {e}")
            finally:
                # Clean up the task when done
                task_manager.cleanup_task(task_id)
                logger.info(f"Task {task_id} cleaned up")
        
        # Start the task thread
        thread = threading.Thread(target=run_task)
        thread.start()
        logger.info(f"Started task {task_id} for action {action}")
        
        return {
            "message": f"Automation '{action}' started",
            "task_id": task_id,
            "status": "running"
        }

    except Exception as e:
        logger.error(f"Error triggering automation for action '{action}': {e}")
        return Response(
            content=f"Error triggering automation for action '{action}': {str(e)}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.post("/stop_task/{task_id}")
def stop_task(task_id: str, user: dict = Depends(get_current_user)):
    """
    Stop a running automation task.
    Staff or admin access required.
    """
    try:
        if task_manager.stop_task(task_id):
            logger.info(f"Task {task_id} is being stopped by user {user['username']}")
            return {"message": f"Task {task_id} is being stopped", "status": "stopping"}
        logger.warning(f"Task {task_id} not found")
        return JSONResponse(
            content={"error": f"Task {task_id} not found"},
            status_code=404
        )
    except Exception as e:
        logger.error(f"Error stopping task {task_id}: {e}")
        return JSONResponse(
            content={"error": f"Error stopping task {task_id}: {str(e)}"},
            status_code=500
        )

@app.get("/task_status/{task_id}")
def get_task_status(task_id: str, user: dict = Depends(get_current_user)):
    """
    Get the status of a specific task.
    Staff or admin access required.
    """
    try:
        status = task_manager.get_task_status(task_id)
        if status:
            return status
        return JSONResponse(
            content={"error": f"Task {task_id} not found"},
            status_code=404
        )
    except Exception as e:
        logger.error(f"Error getting status for task {task_id}: {e}")
        return JSONResponse(
            content={"error": f"Error getting status for task {task_id}: {str(e)}"},
            status_code=500
        )

@app.get("/list_tasks")
def list_tasks(user: dict = Depends(get_current_user)):
    """
    List all running tasks.
    Staff or admin access required.
    """
    try:
        return task_manager.list_tasks()
    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        return JSONResponse(
            content={"error": f"Error listing tasks: {str(e)}"},
            status_code=500
        )

def run_transcription_task(stop_flag):
    """Run the async transcription task in a new event loop"""
    import asyncio
    
    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Run the async function in the new loop
        loop.run_until_complete(get_podcast_audio_transcription(stop_flag))
    except Exception as e:
        logger.error(f"Error in podcast transcription task: {e}")
    finally:
        loop.close()

def run_summary_host_guest_optimized(stop_flag):
    """Run the async optimized summary host guest identification task in a new event loop"""
    import asyncio
    from summary_guest_identification_optimized import PodcastProcessor
    
    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Check if we should stop before starting
        if stop_flag and stop_flag.is_set():
            logger.info("Stop flag set before starting summary_host_guest_optimized")
            return
            
        # Initialize the processor and run the process_all_records method
        # Always use Gemini 2.0 flash without any options to override
        processor = PodcastProcessor()  # PodcastProcessor always uses gemini-2.0-flash by default
        results = loop.run_until_complete(processor.process_all_records(max_concurrency=3, batch_size=5, stop_flag=stop_flag))
        logger.info(f"Optimized summary host guest identification task completed successfully. Processed {results.get('total_processed', 0)} records with {results.get('successful', 0)} successful.")
    except Exception as e:
        logger.error(f"Error in optimized summary host guest identification task: {e}", exc_info=True)
    finally:
        loop.close()


@app.get("/ai-usage")
def get_ai_usage(
    request: Request,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    group_by: str = Query("model", description="Field to group by: 'model', 'workflow', 'endpoint', or 'podcast_id'"),
    format: str = Query("json", description="Output format: 'json', 'text', or 'csv'"),
    user: dict = Depends(get_admin_user)
):
    """
    Get AI usage statistics, optionally filtered by date range.
    Admin access required.
    
    Args:
        start_date: Optional ISO format date (YYYY-MM-DD) to start from
        end_date: Optional ISO format date (YYYY-MM-DD) to end at
        group_by: Field to group results by (model, workflow, endpoint, podcast_id)
        format: Output format (json, text, csv)
        
    Returns:
        Report of AI usage statistics in the requested format
    """
    try:
        report = ai_tracker.generate_report(
            start_date=start_date,
            end_date=end_date,
            group_by=group_by
        )
        
        # Handle different output formats
        if format.lower() == 'text':
            from generate_ai_usage_report import format_as_text
            content = format_as_text(report)
            return Response(content=content, media_type="text/plain")
        
        elif format.lower() == 'csv':
            from generate_ai_usage_report import format_as_csv
            content = format_as_csv(report)
            return Response(content=content, media_type="text/csv", 
                          headers={"Content-Disposition": "attachment; filename=ai_usage_report.csv"})
        
        else:  # Default to json
            return report
            
    except Exception as e:
        logger.error(f"Error generating AI usage report: {e}")
        return JSONResponse(
            content={"error": f"Failed to generate AI usage report: {str(e)}"},
            status_code=500
        )


@app.get("/podcast-cost/{podcast_id}")
def get_podcast_cost(podcast_id: str, user: dict = Depends(get_admin_user)):
    """
    Get detailed AI usage statistics for a specific podcast by its Airtable podcast record ID.
    Admin access required.
    
    This endpoint shows the complete cost analysis for processing a single podcast
    through the entire pipeline from discovery to email creation.
    
    Args:
        podcast_id: The Airtable podcast record ID
        
    Returns:
        JSON report with detailed cost breakdown for this podcast
    """
    try:
        report = ai_tracker.get_record_cost_report(podcast_id)
        return report
    except Exception as e:
        logger.error(f"Error generating podcast cost report for {podcast_id}: {e}")
        return JSONResponse(
            content={"error": f"Failed to generate podcast cost report: {str(e)}"},
            status_code=500
        )


@app.get("/podcast-cost-dashboard/{podcast_id}", response_class=HTMLResponse)
def get_podcast_cost_dashboard(
    request: Request, 
    podcast_id: str, 
    user: dict = Depends(get_admin_user)
):
    """
    Render a dashboard with detailed AI usage statistics for a specific podcast.
    Admin access required.
    
    This endpoint provides a visual dashboard showing the cost analysis for processing 
    a podcast through the entire pipeline.
    
    Args:
        request: The FastAPI request object
        podcast_id: The Airtable podcast record ID
        
    Returns:
        HTML dashboard with detailed cost breakdown for this podcast
    """
    try:
        # Get the report data
        report = ai_tracker.get_record_cost_report(podcast_id)
        
        if "error" in report:
            # Handle case where no data is found
            return HTMLResponse(
                content=f"""
                <html>
                    <head>
                        <title>No Data Found</title>
                        <link rel="stylesheet" href="/static/dashboard.css">
                    </head>
                    <body>
                        <div class="container">
                            <div class="section">
                                <h2>No Data Found</h2>
                                <p>{report["message"]}</p>
                                <a href="/" class="back-button">Back to Dashboard</a>
                            </div>
                        </div>
                    </body>
                </html>
                """,
                status_code=404
            )
        
        # Pass all the report data directly to the template
        return templates.TemplateResponse("podcast_cost.html", {
            "request": request,
            "username": user["username"],
            "role": user["role"],
            **report  # Unpack all report data into the template context
        })
    
    except Exception as e:
        logger.error(f"Error generating podcast cost dashboard for {podcast_id}: {e}")
        return HTMLResponse(
            content=f"""
            <html>
                <head>
                    <title>Error</title>
                    <link rel="stylesheet" href="/static/dashboard.css">
                </head>
                <body>
                    <div class="container">
                        <div class="section">
                            <h2>Error Generating Dashboard</h2>
                            <p>{str(e)}</p>
                            <a href="/" class="back-button">Back to Dashboard</a>
                        </div>
                    </div>
                </body>
            </html>
            """,
            status_code=500
        )


@app.post("/emailsent")
async def webhook_emailsent(request: Request):
    """
    Webhook endpoint that listens for a JSON payload indicating that an email was sent.
    This data is used to update Airtable records via 'update_airtable_when_email_sent'.
    """
    try:
        data = await request.json()
    except Exception:
        logger.warning("No valid JSON received at /emailsent webhook.")
        return JSONResponse(content={"error": "No JSON received"},
                            status_code=400)

    logger.info(f"/emailsent webhook data received: {data}")
    update_airtable_when_email_sent(data)
    return JSONResponse(content={
        "status": "success",
        "message": "Webhook processed!"
    },
                        status_code=200)


@app.post("/replyreceived")
async def webhook_replyreceived(request: Request):
    """
    Webhook endpoint that listens for a JSON payload indicating that a reply was received.
    This data is used to update Airtable records via 'update_correspondent_on_airtable'.
    """
    try:
        data = await request.json()
    except Exception:
        logger.warning("No valid JSON received at /replyreceived webhook.")
        return JSONResponse(content={"error": "No JSON received"},
                            status_code=400)

    logger.info(f"/replyreceived webhook data received: {data}")
    update_correspondent_on_airtable(data)
    return JSONResponse(content={
        "status": "success",
        "message": "Webhook processed!"
    },
                        status_code=200)


@app.get("/transcribe-podcast/{podcast_id}")
async def transcribe_specific_podcast(podcast_id: str, user: dict = Depends(get_admin_user)):
    """
    Endpoint to trigger transcription for a specific podcast by ID.
    Admin access required.
    
    This will look up the podcast in Airtable and transcribe its audio.
    
    Args:
        podcast_id: The Airtable podcast record ID
        
    Returns:
        JSON response with success or error message
    """
    try:
        # Import here to avoid circular imports
        from airtable_service import PodcastService
        
        # Get podcast episode details from Airtable
        airtable = PodcastService()
        record = airtable.get_record("Podcast_Episodes", podcast_id)
        
        if not record:
            logger.error(f"No podcast found with ID: {podcast_id}")
            return JSONResponse(
                content={"error": f"No podcast found with ID: {podcast_id}"},
                status_code=404
            )
        
        audio_url = record.get('fields', {}).get('Episode URL')
        if not audio_url:
            logger.error(f"No audio URL found for podcast ID: {podcast_id}")
            return JSONResponse(
                content={"error": "No audio URL found for this podcast"},
                status_code=400
            )
        
        episode_name = record.get('fields', {}).get('Episode Title', '')
        
        # Start transcription in the background
        import threading
        threading.Thread(
            target=run_specific_transcription,
            args=(podcast_id, audio_url, episode_name)
        ).start()
        
        return {
            "message": f"Transcription started for podcast ID: {podcast_id}",
            "podcast_id": podcast_id,
            "episode_title": episode_name
        }
        
    except Exception as e:
        logger.error(f"Error transcribing podcast {podcast_id}: {e}")
        return JSONResponse(
            content={"error": f"Failed to start transcription: {str(e)}"},
            status_code=500
        )

def run_specific_transcription(podcast_id, audio_url, episode_name):
    """Run the async transcription task for a specific podcast in a new event loop"""
    import asyncio
    
    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Run the async function in the new loop
        transcript_result = loop.run_until_complete(
            transcribe_endpoint(audio_url, episode_name)
        )
        
        # Update the record with the transcript
        from airtable_service import PodcastService
        airtable = PodcastService()
        
        # Update only the two fields that are in the original implementation
        airtable.update_record(
            "Podcast_Episodes", 
            podcast_id, 
            {
                'Transcription': transcript_result.get('transcript', ''),
                'Downloaded': True
            }
        )
        
        logger.info(f"Updated record {podcast_id} with transcription")
        
    except Exception as e:
        logger.error(f"Error in specific podcast transcription task: {e}")
    finally:
        loop.close()


@app.get("/storage-status")
def get_storage_status(user: dict = Depends(get_admin_user)):
    """
    Get detailed information about the AI usage storage system.
    Admin access required.
    
    This helps verify that Replit persistent storage is working correctly.
    """
    try:
        storage_info = ai_tracker.get_storage_info()
        
        # Add some additional environment info
        storage_info.update({
            'replit_info': {
                'REPL_HOME': os.getenv('REPL_HOME', 'Not running on Replit'),
                'REPL_ID': os.getenv('REPL_ID', 'Not available'),
                'REPL_SLUG': os.getenv('REPL_SLUG', 'Not available'),
            },
            'timestamp': datetime.now().isoformat()
        })
        
        return storage_info
    except Exception as e:
        logger.error(f"Error getting storage status: {e}")
        return JSONResponse(
            content={"error": f"Failed to get storage status: {str(e)}"},
            status_code=500
        )


@app.get("/admin")
def admin_dashboard(request: Request, user: dict = Depends(get_admin_user)):
    """
    Admin dashboard page that shows links to all admin-only features.
    Admin access required.
    """
    return templates.TemplateResponse(
        "admin_dashboard.html", 
        {
            "request": request,
            "username": user["username"],
            "role": user["role"]
        }
    )

# Register the test runner routes with the FastAPI app
register_test_routes(app)

# Add a link to the LLM Test Dashboard in the admin template
@app.on_event("startup")
async def startup_event():
    # Update the admin dashboard template to include a link to the LLM Test Dashboard
    try:
        # Just log it for now - in a real app you might want to modify the template
        logger.info("Adding LLM Test Dashboard link to admin dashboard")
    except Exception as e:
        logger.error(f"Error updating admin dashboard: {e}")

# Add a shutdown event handler to perform cleanup
@app.on_event("shutdown")
async def shutdown_event():
    try:
        logger.info("Application shutting down, cleaning up resources...")
        
        # Clean up any running tasks or processes
        # This is especially important for asyncio loops and thread pools
        
        # Close any open database connections or services
        if hasattr(task_manager, 'cleanup'):
            task_manager.cleanup()
        
        # Allow some time for graceful cleanup
        import asyncio
        await asyncio.sleep(0.5)
        
        logger.info("Cleanup completed")
    except Exception as e:
        logger.error(f"Error during application shutdown: {e}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting FastAPI app on port {port}.")

    # Run using uvicorn (most common approach for FastAPI)
    uvicorn.run(app, host='0.0.0.0', port=port)
