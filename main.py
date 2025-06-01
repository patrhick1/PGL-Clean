# main.py (now in the root directory)

import os
import logging
import uuid
import json
from fastapi import FastAPI, Request, Query, Response, status, Form, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager
import sys
import threading

# NEW: APScheduler imports for scheduling
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz # For timezone-aware scheduling
from dotenv import load_dotenv # Ensure dotenv is loaded here as well

# Load environment variables from .env file (important for scheduler config)
load_dotenv()

# Removed: sys.path manipulation - no longer needed if main.py is in root

# Import the authentication middleware (now from src)
from src.auth_middleware import (
    AuthMiddleware,
    authenticate_user,
    create_session,
    get_current_user,
    get_admin_user
)

# Import the task manager (now from src)
from src.task_manager import task_manager

# Import the functions that handle specific tasks (now from src)
from src.webhook_handler import poll_airtable_and_process, poll_podcast_search_database, enrich_host_name
from src.summary_guest_identification_optimized import PodcastProcessor
from src.determine_fit_optimized import determine_fit
from src.pitch_episode_selection_optimized import pitch_episode_selection
from src.pitch_writer_optimized import pitch_writer
from src.send_pitch_to_instantly import send_pitch_to_instantly
from src.instantly_email_sent import update_airtable_when_email_sent
from src.instantly_response import update_correspondent_on_airtable
from src.fetch_episodes import get_podcast_episodes
from src.podcast_note_transcriber import get_podcast_audio_transcription, transcribe_endpoint
from src.free_tier_episode_transcriber import get_podcast_audio_transcription_free_tier, transcribe_endpoint_free_tier

# NEW IMPORT: Import the batch podcast fetcher function (now from src)
from src.batch_podcast_fetcher import process_campaign_keywords

# Import the AI usage tracker (now from src)
from src.ai_usage_tracker import tracker as ai_tracker

# Import your Airtable service classes (now from src)
from src.airtable_service import PodcastService, MIPRService

# Import the database utility functions (now from src)
from src.db_utils import (
    create_history_table,
    get_last_known_value,
    insert_status_history,
    field_value_to_string,
    HISTORY_TABLE_NAME
)

# Import the CampaignStatusTracker (now from src)
from src.campaign_status_tracker import CampaignStatusTracker

# Import necessary functions from instantly_leads_db.py
from src.instantly_leads_db import create_clientsinstantlyleads_table, add_instantly_lead_record
from src.external_api_service import InstantlyAPI # Already imported but good to note dependency
# NEW IMPORT for the deleter script
from src.instantly_lead_deleter import delete_leads_from_instantly, CAMPAIGN_IDS_TO_PROCESS, LEAD_STATUSES_TO_DELETE

# -------------------------------------------------------------------
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# -------------------------------------------------------------------

# Conditionally import and register routes from test_runner.py
ENABLE_LLM_TEST_DASHBOARD = os.getenv("ENABLE_LLM_TEST_DASHBOARD", "false").lower() == "true"

if ENABLE_LLM_TEST_DASHBOARD:
    logger.info("ENABLE_LLM_TEST_DASHBOARD is true. Attempting to load test runner routes.")
    # Adjust path for test_runner if it's not directly in root/tests
    # Assuming tests/test_runner.py is at the same level as src/
    tests_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")

    if not os.path.isdir(tests_dir):
        logger.warning(f"Tests directory not found at {tests_dir}. Cannot load LLM Test Dashboard.")
    else:
        original_sys_path = list(sys.path)
        if tests_dir not in sys.path:
            sys.path.insert(0, tests_dir)
        # No need to insert project_root if main.py is already in root

        try:
            from test_runner import register_routes as register_test_routes # type: ignore
            logger.info("Successfully imported register_routes from test_runner.")
        except ImportError as e:
            logger.error(f"Failed to import test_runner. LLM Test Dashboard will not be available. Error: {e}")
            register_test_routes = None
        except Exception as e:
            logger.error(f"An unexpected error occurred while trying to import test_runner: {e}")
            register_test_routes = None
        finally:
            sys.path = original_sys_path
else:
    logger.info("ENABLE_LLM_TEST_DASHBOARD is false. LLM Test Dashboard routes will not be loaded.")
    register_test_routes = None

# NEW: Initialize APScheduler
app_timezone = os.getenv('APP_TIMEZONE')
if not app_timezone or '#' in app_timezone: # Basic check for comments or empty string
    logger.info(f"APP_TIMEZONE environment variable not set or invalid ('{app_timezone}'), defaulting to UTC for APScheduler.")
    app_timezone = 'UTC'
else:
    logger.info(f"Using APP_TIMEZONE: {app_timezone} for APScheduler.")

try:
    scheduler = BackgroundScheduler(timezone=app_timezone)
except Exception as e:
    logger.warning(f"Failed to initialize APScheduler with timezone '{app_timezone}': {e}. Defaulting to UTC.")
    scheduler = BackgroundScheduler(timezone='UTC')

# NEW: Initialize CampaignStatusTracker globally for the scheduler
# This instance will be used by the scheduled job
tracker_for_scheduler = CampaignStatusTracker()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Application starting up...")

    # --- Database Table Creation on Startup ---
    logger.info("Attempting to create history table...")
    if not create_history_table():
        logger.error("Failed to create history table on startup. Database logging might fail.")
    else:
        logger.info("History table check/creation complete.")
    # --- End Database Table Creation ---

    if ENABLE_LLM_TEST_DASHBOARD and register_test_routes:
        try:
            logger.info("Registering LLM Test Dashboard routes...")
            register_test_routes(app)
            logger.info("LLM Test Dashboard routes registered.")
        except Exception as e:
            logger.error(f"Error registering LLM Test Dashboard routes: {e}")

    # NEW: Schedule the daily report job
    schedule_hour = int(os.getenv('REPORT_SCHEDULE_HOUR', 17)) # Default 5 PM
    schedule_minute = int(os.getenv('REPORT_SCHEDULE_MINUTE', 0)) # Default 0 minutes

    scheduler.add_job(
        run_daily_report_job, # The function to run
        CronTrigger(hour=schedule_hour, minute=schedule_minute),
        id='daily_campaign_report',
        name='Daily Campaign Status Report',
        replace_existing=True
    )
    logger.info(f"Scheduled daily campaign report to run at {schedule_hour:02d}:{schedule_minute:02d} {scheduler.timezone}.")
    
    # Start the scheduler
    scheduler.start()
    logger.info("APScheduler started.")

    yield

    # Shutdown logic
    try:
        logger.info("Application shutting down, cleaning up resources...")
        if hasattr(task_manager, 'cleanup'):
            task_manager.cleanup()
        # NEW: Shut down the scheduler gracefully
        scheduler.shutdown()
        logger.info("APScheduler shut down.")
        import asyncio
        await asyncio.sleep(0.5)
        logger.info("Cleanup completed")
    except Exception as e:
        logger.error(f"Error during application shutdown: {e}")

app = FastAPI(lifespan=lifespan)
app.add_middleware(AuthMiddleware)
templates = Jinja2Templates(directory="templates") # Assumes 'templates' is in the root

def format_number(value):
    return f"{value:,}"

def format_datetime(timestamp):
    dt = datetime.fromisoformat(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

templates.env.filters["format_number"] = format_number
templates.env.filters["format_datetime"] = format_datetime

app.mount("/static", StaticFiles(directory="static"), name="static") # Assumes 'static' is in the root

# --- Dependency for Airtable Service ---
def get_podcast_service() -> PodcastService:
    try:
        return PodcastService()
    except Exception as e:
        logger.critical(f"Failed to initialize PodcastService: {e}")
        raise HTTPException(status_code=500, detail="Internal server error: Airtable service not available.")

# --- Dependency for CampaignStatusTracker ---
# This instance is for webhook/manual triggers, not the scheduled job
def get_campaign_status_tracker() -> CampaignStatusTracker:
    try:
        return CampaignStatusTracker()
    except Exception as e:
        logger.critical(f"Failed to initialize CampaignStatusTracker: {e}")
        raise HTTPException(status_code=500, detail="Internal server error: Campaign status tracker not available.")

# NEW: Function to be scheduled by APScheduler
def run_daily_report_job():
    """Function to be scheduled by APScheduler."""
    logger.info("Scheduler triggered: Running daily campaign status report.")
    try:
        # Use the globally initialized tracker instance
        tracker_for_scheduler.update_all_client_spreadsheets()
        logger.info("Daily campaign status report job completed.")
    except Exception as e:
        logger.error(f"Error during daily campaign status report job: {e}", exc_info=True)

@app.get("/login")
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...)
):
    role = authenticate_user(username, password)
    if not role:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error": "Invalid username or password"
            }
        )
    session_id = create_session(username, role)
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=3600
    )
    return response


@app.get("/logout")
def logout(request: Request):
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(key="session_id")
    return response


@app.get("/")
def index(request: Request, user: dict = Depends(get_current_user)):
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
    return {"message": "PGL Automation API is running (FastAPI version)!"}


# --- NEW Airtable Webhook Endpoint for Status Changes ---
@app.post("/webhook/airtable-status-change")
async def handle_airtable_status_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    podcast_service: PodcastService = Depends(get_podcast_service),
    campaign_tracker: CampaignStatusTracker = Depends(get_campaign_status_tracker) # This is the instance for webhook
):
    """
    Receives and processes Airtable webhook payloads for status changes in Campaign Manager.
    Logs status changes to the PostgreSQL history table and triggers spreadsheet updates.
    """
    try:
        payload = await request.json()
        logger.info(f"Received Airtable webhook payload: {json.dumps(payload, indent=2)}")

        if not payload or 'changedTablesById' not in payload:
            logger.warning("Invalid webhook payload received: Missing 'changedTablesById'.")
            raise HTTPException(status_code=400, detail="Invalid webhook payload.")

        base_id = payload.get('base', {}).get('id')
        if not base_id:
            logger.error("Webhook payload missing base ID.")
            raise HTTPException(status_code=400, detail="Missing base ID in webhook payload.")

        campaign_manager_table_id = os.getenv('CAMPAIGN_MANAGER_TABLE_ID', 'tblFEQc6i54amoxBM')
        campaign_manager_table_name = 'Campaign Manager'

        table_changes = payload['changedTablesById'].get(campaign_manager_table_id)

        if not table_changes or 'changedRecords' not in table_changes:
            logger.info(f"No relevant changes for table '{campaign_manager_table_name}' ({campaign_manager_table_id}) in this webhook.")
            return {"message": "No relevant changes processed."}

        for record_change in table_changes['changedRecords']:
            record_id = record_change.get('id')
            if not record_id:
                logger.warning(f"Skipping record change due to missing record ID in table {campaign_manager_table_id}.")
                continue

            current_fields = record_change.get('current', {}).get('fields', {})
            previous_fields = record_change.get('previous', {}).get('fields', {})

            if 'Status' in current_fields:
                current_status = current_fields['Status']
                old_status = previous_fields.get('Status')

                if old_status is None:
                    logger.debug(f"Old status not in webhook for {record_id}. Fetching from DB history.")
                    old_status = get_last_known_value(
                        airtable_record_id=record_id,
                        airtable_table_name=campaign_manager_table_name,
                        airtable_base_id=base_id,
                        field_name='Status'
                    )
                    if old_status is None:
                        old_status = "Initial"

                if old_status != current_status:
                    logger.info(f"Detected status change for record {record_id}: '{old_status}' -> '{current_status}'.")

                    # Fetch full record details from Airtable for denormalized fields
                    full_airtable_record = podcast_service.get_record(campaign_manager_table_name, record_id)
                    if full_airtable_record:
                        full_fields = full_airtable_record.get('fields', {})
                        client_name = field_value_to_string(full_fields.get('Client Name'))
                        campaign_name = field_value_to_string(full_fields.get('CampaignName'))
                        podcast_name = field_value_to_string(full_fields.get('Podcast'))
                    else:
                        logger.warning(f"Could not fetch full Airtable record {record_id} for denormalized fields. Denormalized fields will be None.")
                        client_name = None
                        campaign_name = None
                        podcast_name = None

                    # Log to PostgreSQL history table
                    insert_status_history(
                        airtable_record_id=record_id,
                        airtable_table_name=campaign_manager_table_name,
                        airtable_base_id=base_id,
                        field_name='Status',
                        old_value=old_status,
                        new_value=current_status,
                        client_name=client_name,
                        campaign_name=campaign_name,
                        podcast_name=podcast_name,
                        source_system='Airtable Webhook'
                    )

                    # Trigger Google Sheet Update in Background for the specific client
                    if client_name:
                        logger.info(f"Adding background task to update Google Sheet for client: {client_name}")
                        background_tasks.add_task(campaign_tracker.update_single_client_spreadsheet, client_name)
                    else:
                        logger.warning(f"Client name not found for record {record_id}. Cannot update client spreadsheet.")

                else:
                    logger.info(f"No actual status change detected for record {record_id} ('{current_status}' == '{old_status}'). Not logging or updating sheet.")
            else:
                logger.debug(f"Status field not in current fields for record {record_id}. Skipping.")

        return {"message": "Webhook received and processed successfully."}

    except json.JSONDecodeError:
        logger.error("Received non-JSON payload at /webhook/airtable-status-change.")
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing Airtable status webhook: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
# --- END NEW Airtable Webhook Endpoint ---


@app.get("/trigger_automation")
def trigger_automation(
        action: str = Query(...,
                            description="Name of the automation to trigger"),
        id: Optional[str] = Query(
            None, description="Record ID for the automation if needed")):
    try:
        task_id = str(uuid.uuid4())
        task_manager.start_task(task_id, action)
        def run_task():
            try:
                stop_flag = task_manager.get_stop_flag(task_id)
                if not stop_flag:
                    logger.error(f"Could not get stop flag for task {task_id}")
                    return
                if action == 'generate_bio_angles':
                    if not id:
                        logger.warning("No record ID provided for generate_bio_angles, will process all eligible records")
                    poll_airtable_and_process(id, stop_flag)
                elif action == 'mipr_podcast_search':
                    if not id:
                        raise ValueError("Missing 'id' parameter for MIPR Podcast Search automation!")
                    poll_podcast_search_database(id, stop_flag)
                elif action == 'fetch_podcast_episodes':
                    get_podcast_episodes(stop_flag)
                elif action == 'summary_host_guest':
                    run_summary_host_guest_optimized(stop_flag)
                elif action == 'determine_fit':
                    determine_fit(stop_flag, batch_size=5)
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
                elif action == 'transcribe_podcast_free_tier':
                    run_transcription_task_free_tier(stop_flag)
                elif action == 'update_all_client_spreadsheets': # NEW: Add action for full sheet refresh
                    # For manual trigger, create a new instance as it runs in a separate thread
                    campaign_tracker_instance = CampaignStatusTracker()
                    campaign_tracker_instance.update_all_client_spreadsheets()
                # NEW ACTION: Trigger batch podcast fetching for a campaign
                elif action == 'batch_podcast_fetch':
                    if not id:
                        raise ValueError("Missing 'id' parameter for Batch Podcast Fetch automation (Airtable Campaign Record ID)!")
                    logger.info(f"Starting batch podcast fetch for campaign record ID: {id}")
                    # Run the batch process in a separate thread
                    threading.Thread(
                        target=process_campaign_keywords,
                        args=(id, stop_flag)
                    ).start()
                    return {"message": f"Batch podcast fetch started for Campaign ID '{id}'.", "task_id": task_id, "status": "running"}
                else:
                    raise ValueError(f"Invalid action: {action}")
            except Exception as e:
                logger.error(f"Error in task {task_id}: {e}")
            finally:
                task_manager.cleanup_task(task_id)
                logger.info(f"Task {task_id} cleaned up")
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

@app.post("/trigger-daily-report")
async def trigger_daily_report_manually(
    # CORRECTED: BackgroundTasks should be injected as a dependency
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_admin_user)
):
    """
    Endpoint to manually trigger the daily campaign status report.
    This runs the report in a background task to avoid blocking the API response.
    Requires admin privileges.
    """
    logger.info(f"Manual trigger for daily campaign status report received by user: {user['username']}.")
    try:
        # Add the job to background tasks. FastAPI will manage its execution
        # without blocking the HTTP response.
        background_tasks.add_task(run_daily_report_job)
        return {"message": "Daily campaign status report triggered successfully. It is running in the background.", "status": "processing"}
    except Exception as e:
        logger.error(f"Error triggering manual daily report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger report: {e}")


@app.post("/stop_task/{task_id}")
def stop_task(task_id: str, user: dict = Depends(get_current_user)):
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
    try:
        return task_manager.list_tasks()
    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        return JSONResponse(
            content={"error": f"Error listing tasks: {str(e)}"},
            status_code=500
        )

def run_transcription_task(stop_flag):
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Corrected: Should call the main batch transcription function
        from src.podcast_note_transcriber import get_podcast_audio_transcription # Ensure this is the correct function name
        loop.run_until_complete(get_podcast_audio_transcription(stop_flag))
    except Exception as e:
        logger.error(f"Error in podcast transcription task: {e}")
    finally:
        loop.close()

def run_transcription_task_free_tier(stop_flag):
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Corrected: Should call the main batch transcription function for free tier
        from src.free_tier_episode_transcriber import get_podcast_audio_transcription_free_tier # Ensure this is the correct function name
        loop.run_until_complete(get_podcast_audio_transcription_free_tier(stop_flag))
    except Exception as e:
        logger.error(f"Error in podcast transcription task (free tier): {e}") # Differentiated log message
    finally:
        loop.close()

def run_summary_host_guest_optimized(stop_flag):
    import asyncio
    from src.summary_guest_identification_optimized import PodcastProcessor # Updated import
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        if stop_flag and stop_flag.is_set():
            logger.info("Stop flag set before starting summary_host_guest_optimized")
            return
        processor = PodcastProcessor()
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
    try:
        from src.generate_ai_usage_report import format_as_text, format_as_csv # Moved import inside function
        report = ai_tracker.generate_report(
            start_date=start_date,
            end_date=end_date,
            group_by=group_by
        )
        if format.lower() == 'text':
            content = format_as_text(report)
            return Response(content=content, media_type="text/plain")
        elif format.lower() == 'csv':
            content = format_as_csv(report)
            return Response(content=content, media_type="text/csv",
                          headers={"Content-Disposition": "attachment; filename=ai_usage_report.csv"})
        else:
            return report
    except Exception as e:
        logger.error(f"Error generating AI usage report: {e}")
        return JSONResponse(
            content={"error": f"Failed to generate AI usage report: {str(e)}"},
            status_code=500
        )


@app.get("/podcast-cost/{podcast_id}")
def get_podcast_cost(podcast_id: str, user: dict = Depends(get_admin_user)):
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
    try:
        report = ai_tracker.get_record_cost_report(podcast_id)
        if "error" in report:
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
        return templates.TemplateResponse("podcast_cost.html", {
            "request": request,
            "username": user["username"],
            "role": user["role"],
            **report
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
    try:
        from src.airtable_service import PodcastService # Updated import
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
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Assuming transcribe_endpoint_free_tier is imported from src.free_tier_episode_transcriber
        from src.free_tier_episode_transcriber import transcribe_endpoint_free_tier
        transcript_result = loop.run_until_complete(
            transcribe_endpoint_free_tier(audio_url, episode_name)
        )
        from src.airtable_service import PodcastService # Updated import
        airtable = PodcastService()
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
    try:
        storage_info = ai_tracker.get_storage_info()
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
    return templates.TemplateResponse(
        "admin_dashboard.html",
        {
            "request": request,
            "username": user["username"],
            "role": user["role"]
        }
    )

# --- Instantly.ai Leads Backup --- 
# Define campaign IDs for backup (can be moved to config or env vars later)
ALL_INSTANTLY_CAMPAIGN_IDS = [
    "afe3a4d7-5ed7-4fd4-9f8f-cf4e2ddc843d",
    "d52f85c0-8341-42d8-9e07-99c6b758fa0b",
    "7b4a5386-8fa1-4059-8ded-398c0f48972b",
    "186fcab7-7c86-4086-9278-99238c453470",
    "ae1c1042-d10e-4cfc-ba4c-743a42550c85",
    "ccbd7662-bbed-46ee-bd8f-1bc374646472",
    "ad2c89bc-686d-401e-9f06-c6ff9d9b7430",
    # "3816b624-2a1f-408e-91a9-b9f730d03e2b", # This one will be excluded
    "60346de6-915c-43fa-9dfa-b77983570359",
    "5b1053b5-8143-4814-a9dc-15408971eac8",
    "02b1d9ff-0afe-4b64-ac15-a886f43bdbce",
    "0725cdd8-b090-4da4-90af-6ca93ac3c267",
    "640a6822-c1a7-48c7-8385-63b0d4c283fc",
    "540b0539-f1c2-4612-94d8-df6fab42c2a7",
    "b55c61b6-262c-4390-b6e0-63dfca1620c2"
]
EXCLUDED_INSTANTLY_CAMPAIGN_ID = "3816b624-2a1f-408e-91a9-b9f730d03e2b"

def run_instantly_leads_backup_job(stop_flag: Optional[threading.Event] = None):
    """Function to run the Instantly.ai leads backup process."""
    logger.info("Starting Instantly.ai leads backup job...")
    try:
        create_clientsinstantlyleads_table() # Ensure table exists

        campaign_ids_to_backup = [cid for cid in ALL_INSTANTLY_CAMPAIGN_IDS if cid != EXCLUDED_INSTANTLY_CAMPAIGN_ID]

        if not campaign_ids_to_backup:
            logger.info("No campaign IDs specified for Instantly backup. Skipping.")
            return

        logger.info(f"--- Starting Lead Backup for {len(campaign_ids_to_backup)} Campaign(s) ---")
        instantly_service = InstantlyAPI()
        total_leads_fetched_all_campaigns = 0
        total_leads_added_all_campaigns = 0
        total_leads_failed_all_campaigns = 0

        for campaign_idx, current_campaign_id in enumerate(campaign_ids_to_backup):
            if stop_flag and stop_flag.is_set():
                logger.info("Instantly leads backup job stopped by flag.")
                break
            logger.info(f"Processing Campaign {campaign_idx + 1}/{len(campaign_ids_to_backup)}: ID {current_campaign_id}")
            
            leads_from_api = instantly_service.list_leads_from_campaign(current_campaign_id)

            if leads_from_api:
                logger.info(f"Fetched {len(leads_from_api)} leads from Instantly API for campaign {current_campaign_id}.")
                total_leads_fetched_all_campaigns += len(leads_from_api)
                current_campaign_added_count = 0
                current_campaign_failed_count = 0
                
                for i, lead_data in enumerate(leads_from_api):
                    if stop_flag and stop_flag.is_set():
                        logger.info("Instantly leads backup for current campaign stopped by flag.")
                        break
                    if (i + 1) % 100 == 0 or i == len(leads_from_api) - 1:
                         logger.info(f"  Processing lead {i+1}/{len(leads_from_api)} for campaign {current_campaign_id}...")
                    
                    inserted_id = add_instantly_lead_record(lead_data)
                    if inserted_id:
                        current_campaign_added_count += 1
                    else:
                        current_campaign_failed_count += 1
                
                total_leads_added_all_campaigns += current_campaign_added_count
                total_leads_failed_all_campaigns += current_campaign_failed_count
                logger.info(f"Backup for campaign {current_campaign_id} complete.")
                logger.info(f"  Successfully processed/upserted: {current_campaign_added_count} lead(s).")
                if current_campaign_failed_count > 0:
                    logger.info(f"  Failed to process/upsert: {current_campaign_failed_count} lead(s).")
            elif isinstance(leads_from_api, list) and not leads_from_api:
                logger.info(f"No leads found in Instantly campaign {current_campaign_id} or an API error occurred.")
            else:
                logger.warning(f"Could not fetch leads from API for campaign {current_campaign_id}. Unexpected return type or error.")
            if stop_flag and stop_flag.is_set(): break # break outer loop too
        
        logger.info("--- Overall Instantly Leads Backup Summary ---")
        logger.info(f"Total campaigns processed: {len(campaign_ids_to_backup) if not (stop_flag and stop_flag.is_set()) else campaign_idx + 1}")
        logger.info(f"Total leads fetched from API across all processed campaigns: {total_leads_fetched_all_campaigns}")
        logger.info(f"Total leads successfully added/updated in DB: {total_leads_added_all_campaigns}")
        if total_leads_failed_all_campaigns > 0:
            logger.info(f"Total leads failed to add/update in DB: {total_leads_failed_all_campaigns}")
        logger.info("Instantly.ai leads backup job finished.")

    except Exception as e:
        logger.error(f"Error during Instantly.ai leads backup job: {e}", exc_info=True)

@app.post("/trigger-instantly-backup")
async def trigger_instantly_backup_manually(
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_admin_user) # Or get_current_user if staff can also trigger
):
    """Endpoint to manually trigger the Instantly.ai leads backup."""
    logger.info(f"Manual trigger for Instantly.ai leads backup received by user: {user['username']}.")
    task_id = str(uuid.uuid4()) # Generate a unique task ID
    # We can use the existing task_manager if we want to make this stoppable
    # For now, just running in background_tasks for simplicity
    background_tasks.add_task(run_instantly_leads_backup_job) # No stop_flag passed for simple background task
    return {"message": "Instantly.ai leads backup job started in the background.", "task_id": task_id, "status": "processing"}

# --- End Instantly.ai Leads Backup ---

# --- Instantly.ai Leads Deletion ---
def run_instantly_leads_deletion_job(stop_flag: Optional[threading.Event] = None):
    """Function to run the Instantly.ai leads deletion process."""
    logger.info("Starting Instantly.ai leads deletion job...")
    try:
        # Ensure the main table exists (though deleter doesn't write to it, it reads via get_instantly_lead_by_id)
        create_clientsinstantlyleads_table() 
        # Run the deletion process (defaults to dry_run=True for safety)
        # Pass the globally defined CAMPAIGN_IDS_TO_PROCESS and LEAD_STATUSES_TO_DELETE
        # You might want to make dry_run configurable via endpoint parameter or env var for actual deletion.
        delete_leads_from_instantly(
            campaign_ids=CAMPAIGN_IDS_TO_PROCESS, 
            statuses_to_delete=LEAD_STATUSES_TO_DELETE, 
            dry_run=False # IMPORTANT: Changed to False for actual deletion via API
        )
        logger.info("Instantly.ai leads deletion job finished.")
    except Exception as e:
        logger.error(f"Error during Instantly.ai leads deletion job: {e}", exc_info=True)

@app.post("/trigger-instantly-lead-deletion")
async def trigger_instantly_lead_deletion_manually(
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_admin_user) 
):
    """Endpoint to manually trigger the Instantly.ai leads deletion."""
    logger.info(f"Manual trigger for Instantly.ai leads deletion received by user: {user['username']}.")
    # For now, running in background_tasks. If it becomes very long, consider TaskManager.
    background_tasks.add_task(run_instantly_leads_deletion_job)
    return {"message": "Instantly.ai leads deletion job started in the background (Dry Run Mode by default).", "status": "processing"}
# --- End Instantly.ai Leads Deletion ---

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting FastAPI app on port {port}.")
    # Changed from "src.main_fastapi:app" to "main:app"
    uvicorn.run("main:app", host='0.0.0.0', port=port, reload=True)