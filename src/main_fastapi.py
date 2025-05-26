# main_fastapi.py

import os
import logging
import uuid
import json
from fastapi import FastAPI, Request, Query, Response, status, Form, Depends, HTTPException, BackgroundTasks # Added BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager
import sys
import threading

# Add the project root directory to sys.path to allow importing 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # Go up one level from 'src'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the authentication middleware
from .auth_middleware import (
    AuthMiddleware,
    authenticate_user,
    create_session,
    get_current_user,
    get_admin_user
)

# Import the task manager
from .task_manager import task_manager

# Import the functions that handle specific tasks
from .webhook_handler import poll_airtable_and_process, poll_podcast_search_database, enrich_host_name
from .summary_guest_identification_optimized import PodcastProcessor
from .determine_fit_optimized import determine_fit
from .pitch_episode_selection_optimized import pitch_episode_selection
from .pitch_writer_optimized import pitch_writer
from .send_pitch_to_instantly import send_pitch_to_instantly
from .instantly_email_sent import update_airtable_when_email_sent
from .instantly_response import update_correspondent_on_airtable
from .fetch_episodes import get_podcast_episodes
from .podcast_note_transcriber import get_podcast_audio_transcription, transcribe_endpoint
from .free_tier_episode_transcriber import get_podcast_audio_transcription_free_tier, transcribe_endpoint_free_tier

# NEW IMPORT: Import the batch podcast fetcher function
from .batch_podcast_fetcher import process_campaign_keywords # NEW IMPORT

# Import the AI usage tracker
from .ai_usage_tracker import tracker as ai_tracker

# Import your Airtable service classes
from .airtable_service import PodcastService, MIPRService

# Import the database utility functions
from .db_utils import (
    create_history_table,
    get_last_known_value,
    insert_status_history,
    field_value_to_string,
    HISTORY_TABLE_NAME
)

# Import the CampaignStatusTracker
from .campaign_status_tracker import CampaignStatusTracker # NEW IMPORT

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
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tests_dir = os.path.join(project_root, "tests")

    if not os.path.isdir(tests_dir):
        logger.warning(f"Tests directory not found at {tests_dir}. Cannot load LLM Test Dashboard.")
    else:
        original_sys_path = list(sys.path)
        if tests_dir not in sys.path:
            sys.path.insert(0, tests_dir)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Application starting up...")

    # --- Database Table Creation on Startup ---
    logger.info("Attempting to create history table...")
    if not create_history_table():
        logger.error("Failed to create history table on startup. Database logging might fail.")
        # Depending on criticality, you might want to raise an exception here
        # to prevent the app from starting if DB is essential.
        # raise RuntimeError("Database table creation failed.")
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

    yield

    # Shutdown logic
    try:
        logger.info("Application shutting down, cleaning up resources...")
        if hasattr(task_manager, 'cleanup'):
            task_manager.cleanup()
        import asyncio
        await asyncio.sleep(0.5)
        logger.info("Cleanup completed")
    except Exception as e:
        logger.error(f"Error during application shutdown: {e}")

app = FastAPI(lifespan=lifespan)
app.add_middleware(AuthMiddleware)
templates = Jinja2Templates(directory="templates")

def format_number(value):
    return f"{value:,}"

def format_datetime(timestamp):
    dt = datetime.fromisoformat(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

templates.env.filters["format_number"] = format_number
templates.env.filters["format_datetime"] = format_datetime

app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Dependency for Airtable Service ---
def get_podcast_service() -> PodcastService:
    try:
        return PodcastService()
    except Exception as e:
        logger.critical(f"Failed to initialize PodcastService: {e}")
        raise HTTPException(status_code=500, detail="Internal server error: Airtable service not available.")

# --- Dependency for CampaignStatusTracker ---
def get_campaign_status_tracker() -> CampaignStatusTracker:
    try:
        return CampaignStatusTracker()
    except Exception as e:
        logger.critical(f"Failed to initialize CampaignStatusTracker: {e}")
        raise HTTPException(status_code=500, detail="Internal server error: Campaign status tracker not available.")


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
    background_tasks: BackgroundTasks, # NEW: Added BackgroundTasks
    podcast_service: PodcastService = Depends(get_podcast_service),
    campaign_tracker: CampaignStatusTracker = Depends(get_campaign_status_tracker) # NEW: Added CampaignStatusTracker
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

                    # --- NEW: Trigger Google Sheet Update in Background ---
                    if client_name:
                        logger.info(f"Adding background task to update Google Sheet for client: {client_name}")
                        background_tasks.add_task(campaign_tracker.update_single_client_spreadsheet, client_name)
                    else:
                        logger.warning(f"Client name not found for record {record_id}. Cannot update client spreadsheet.")
                    # --- END NEW ---

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
                    campaign_tracker_instance = CampaignStatusTracker() # Create new instance for background task
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
        loop.run_until_complete(get_podcast_audio_transcription_free_tier(stop_flag))
    except Exception as e:
        logger.error(f"Error in podcast transcription task: {e}")
    finally:
        loop.close()

def run_summary_host_guest_optimized(stop_flag):
    import asyncio
    from .summary_guest_identification_optimized import PodcastProcessor
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
        report = ai_tracker.generate_report(
            start_date=start_date,
            end_date=end_date,
            group_by=group_by
        )
        if format.lower() == 'text':
            from .generate_ai_usage_report import format_as_text
            content = format_as_text(report)
            return Response(content=content, media_type="text/plain")
        elif format.lower() == 'csv':
            from .generate_ai_usage_report import format_as_csv
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
        from .airtable_service import PodcastService
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
        transcript_result = loop.run_until_complete(
            transcribe_endpoint_free_tier(audio_url, episode_name)
        )
        from .airtable_service import PodcastService
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting FastAPI app on port {port}.")
    uvicorn.run("src.main_fastapi:app", host='0.0.0.0', port=port, reload=True)