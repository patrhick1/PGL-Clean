# PGL (Podcast Guest Logistics) Automation Platform

An advanced automation platform for podcast guest management, outreach, and analytics.

## Overview

This platform automates the entire podcast guest booking workflow:
- Discovering and analyzing podcasts
- Identifying potential guests and hosts
- Evaluating podcast-guest fit
- Crafting personalized pitches
- Sending and tracking outreach
- Analyzing responses and success rates
- Comprehensive AI usage tracking and cost analysis

## Key Features

- **Podcast Discovery & Analysis**: Automatically fetch podcast episodes and transcribe content
- **Guest/Host Identification**: Extract and enrich data about hosts and guests
- **Fit Determination**: AI-powered analysis to match guests with relevant podcasts  
- **Automated Pitching**: Generate personalized pitches with tailored angles and content
- **Email Integration**: Send pitches via third-party email services
- **Usage Analytics**: Track AI model usage, token consumption, and costs
- **Admin Dashboard**: Monitor workflows, track tasks, and view system performance
- **Authentication System**: Secure user login with role-based permissions
- **Scheduled Reporting**: Automated daily/weekly reports to Slack and Google Sheets.

## System Requirements

- Python 3.9+
- FastAPI
- PostgreSQL
- Various AI model APIs (OpenAI, Anthropic, Google)
- Airtable for data storage
- Google Workspace (Docs, Sheets, Drive APIs)

## Getting Started

1. **Create a virtual environment:**
   ```
   python -m venv pgl_env
   ```

2. **Activate the virtual environment:**
   - Windows: `.\pgl_env\Scripts\activate`
   - Mac/Linux: `source pgl_env/bin/activate`

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Create a `.env` file with required API keys and configuration:**
   ```
   # AI Model API Keys
   ANTHROPIC_API=your_anthropic_api_key
   OPENAI_API=your_openai_api_key
   GEMINI_API_KEY=your_gemini_api_key
   
   # Airtable Configuration
   AIRTABLE_PERSONAL_TOKEN=your_airtable_personal_token # Or AIRTABLE_API_KEY for older setups
   PODCAST_BASE_ID=your_podcast_base_id
   MIPR_CRM_BASE_ID=your_mipr_crm_base_id # If using MIPRService
   CAMPAIGN_MANAGER_TABLE_ID=your_campaign_manager_table_id # For webhook
   
   # Email Service API (Instantly.ai)
   INSTANTLY_API_KEY=your_instantly_api_key

   # Google Cloud / Workspace
   GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json
   GOOGLE_PODCAST_INFO_FOLDER_ID=your_google_drive_folder_id_for_podcast_info_docs
   CLIENT_SPREADSHEETS_TRACKING_FOLDER_ID=your_google_drive_folder_id_for_client_reports

   # PostgreSQL Database Connection
   PGDATABASE=your_db_name
   PGUSER=your_db_user
   PGPASSWORD=your_db_password
   PGHOST=your_db_host
   PGPORT=your_db_port # e.g., 5432

   # Slack Webhook for Notifications
   SLACK_WEBHOOK_URL=your_slack_webhook_url

   # Scheduler Configuration
   APP_TIMEZONE=UTC # e.g., America/New_York, Europe/London
   REPORT_SCHEDULE_HOUR=9 # e.g., 9 for 9 AM
   REPORT_SCHEDULE_MINUTE=0
   
   # Authentication (configure as needed)
   # Passwords for ADMIN_USERS and STAFF_USERS are set directly in src/auth_middleware.py
   # Consider moving to environment variables for better security.

   # Optional: Enable LLM Test Dashboard
   ENABLE_LLM_TEST_DASHBOARD=false # Set to true to enable
   ```

## Application Structure

- `main.py`: Main application entry point and API routes (FastAPI)
- `src/`: Directory containing all core application modules.
    - `auth_middleware.py`: Authentication and session management
    - `task_manager.py`: Background task management and monitoring
    - `ai_usage_tracker.py`: AI model usage and cost tracking
    - `webhook_handler.py`: Processes Airtable webhooks for generating bios/angles
    - `airtable_service.py`: Airtable data access and manipulation
    - `batch_podcast_fetcher.py`: Logic for fetching podcasts based on campaign keywords
    - `campaign_status_tracker.py`: Generates client campaign reports (Sheets, Slack)
    - `config.py`: Configuration constants for reporting and other modules
    - `data_processor.py`: Utility functions for processing data from APIs
    - `db_utils.py`: PostgreSQL database interaction utilities
    - `external_api_service.py`: Clients for external APIs (ListenNotes, Podscan, Instantly)
    - `fetch_episodes.py`: Fetches and manages podcast episodes
    - `free_tier_episode_transcriber.py`: Transcription using free tier/alternative services
    - `gemini_service.py` / `openai_service.py` / `anthropic_service.py`: AI model interaction services
    - `google_docs_service.py` / `google_sheets_service.py`: Google Workspace API interactions
    - `instantly_leads_db.py`: Manages local backup of Instantly.ai leads
    - `slack_service.py`: Sends notifications and reports to Slack
    - ... and other specialized modules for specific automation tasks.
- `templates/`: HTML templates for web interface (Jinja2)
- `static/`: CSS, JS, and other static assets
- `prompts/`: AI prompt templates
- `tests/`: Unit and integration tests (if `ENABLE_LLM_TEST_DASHBOARD=true`)

## Automation Workflows

The platform provides several automation workflows accessible via API endpoints (primarily through the dashboard):

### Podcast Discovery & Processing
- `/trigger_automation?action=batch_podcast_fetch&id={campaign_record_id}`: Fetches podcasts based on campaign keywords.
- `/trigger_automation?action=fetch_podcast_episodes`: Retrieves and updates episodes for existing podcasts.
- `/trigger_automation?action=summary_host_guest`: Processes podcast summaries to identify hosts and guests.
- `/transcribe-podcast/{podcast_episode_id}`: Transcribes a specific podcast episode.
- `/trigger_automation?action=transcribe_podcast`: Transcribes all pending episodes (paid tier).
- `/trigger_automation?action=transcribe_podcast_free_tier`: Transcribes all pending episodes (free tier).

### Client & Campaign Management
- `/trigger_automation?action=generate_bio_angles`: Generates guest bios and angles based on Google Docs content.
- `/trigger_automation?action=mipr_podcast_search&id={mipr_record_id}`: Initiates a podcast search based on an MIPR CRM record.

### Guest Matching & Pitching
- `/trigger_automation?action=enrich_host_name`: Enhances host data with additional information.
- `/trigger_automation?action=determine_fit`: Evaluates guest-podcast compatibility.
- `/trigger_automation?action=pitch_episode_angle`: Selects optimal episodes and develops pitch angles.
- `/trigger_automation?action=pitch_writer`: Generates customized pitch email content.
- `/trigger_automation?action=send_pitch`: Sends generated pitches via Instantly.ai.

### Reporting & Data Management
- `POST /trigger-daily-report`: Manually triggers the generation and sending of daily campaign status reports.
- `POST /trigger-instantly-backup`: Manually triggers the backup of leads from Instantly.ai to the local database.
- `/trigger_automation?action=update_all_client_spreadsheets`: Manually refreshes all client campaign status spreadsheets.

## Task Management

Monitor and control long-running automation tasks with these endpoints (typically via the dashboard):

- `/list_tasks`: View all running automation tasks.
- `/task_status/{task_id}`: Check the status of a specific task.
- `/stop_task/{task_id}`: Stop a running task (requires authentication).

## AI Usage Analytics

### Web Interface

Access the built-in reporting dashboard (requires admin privileges):
```
GET /ai-usage
```
Query parameters:
- `start_date`: Filter from this date (YYYY-MM-DD)
- `end_date`: Filter to this date (YYYY-MM-DD)
- `group_by`: Group results by "model", "workflow", "endpoint", or "podcast_id"
- `format`: Output format ("json", "text", or "csv")

### Per-Podcast Analytics

View costs for specific podcasts (requires admin privileges):
```
GET /podcast-cost/{podcast_id}
GET /podcast-cost-dashboard/{podcast_id}
```

### Command-Line Reporting

Generate detailed reports (run from the `src` directory):
```
python generate_ai_usage_report.py --format json --output ../report.json
```

For comprehensive AI usage tracking documentation, see [docs/AI_USAGE_TRACKING.md](docs/AI_USAGE_TRACKING.md).

## Running the Application

Ensure your `.env` file is correctly configured in the project root directory.

Start the FastAPI server (from the project root directory):
```
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The application will be available at: http://localhost:8000

## Authentication

- `/login`: User authentication page.
- `/logout`: End user session.
- `/admin`: Admin dashboard (requires admin privileges).

## API Documentation

FastAPI automatically generates interactive API documentation when the application is running:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
