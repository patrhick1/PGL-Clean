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

## System Requirements

- Python 3.9+
- FastAPI
- Various AI model APIs (OpenAI, Anthropic, Google)
- Airtable for data storage

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

4. **Create a `.env` file with required API keys:**
   ```
   # AI Model API Keys
   ANTHROPIC_API=your_anthropic_api_key
   OPENAI_API=your_openai_api_key
   GEMINI_API=your_gemini_api_key
   
   # Airtable Configuration
   AIRTABLE_API_KEY=your_airtable_api_key
   AIRTABLE_BASE_ID=your_airtable_base_id
   
   # Email Service API (if using)
   INSTANTLY_API_KEY=your_instantly_api_key
   
   # Authentication (configure as needed)
   ADMIN_USERNAME=admin_username
   ADMIN_PASSWORD=admin_password
   USER_USERNAME=user_username
   USER_PASSWORD=user_password
   ```

## Application Structure

- `main_fastapi.py`: Main application entry point and API routes
- `auth_middleware.py`: Authentication and session management
- `task_manager.py`: Background task management and monitoring
- `ai_usage_tracker.py`: AI model usage and cost tracking
- `webhook_handler.py`: Process external service webhooks
- `airtable_service.py`: Airtable data access and manipulation
- `/templates`: HTML templates for web interface
- `/static`: CSS, JS, and static assets
- `/prompts`: AI prompt templates and configurations

## Automation Workflows

The platform provides several automation workflows accessible via API endpoints:

### Podcast Processing

- `/trigger_automation?action=fetch_podcast_episodes`: Retrieve episodes from podcast feeds
- `/trigger_automation?action=summary_host_guest`: Process podcast summaries and identify hosts and guests
- `/transcribe-podcast/{podcast_id}`: Transcribe podcast audio content

### Guest Matching

- `/trigger_automation?action=enrich_host_name`: Enhance host data with additional information
- `/trigger_automation?action=determine_fit`: Evaluate guest-podcast compatibility 

### Pitch Generation and Sending

- `/trigger_automation?action=pitch_episode_angle`: Select optimal episodes and develop angles for pitches
- `/trigger_automation?action=pitch_writer`: Generate customized pitch content
- `/trigger_automation?action=send_pitch`: Send pitches via email service

### Additional Workflows

- `/trigger_automation?action=generate_bio_angles`: Generate guest bios and angles
- `/trigger_automation?action=mipr_podcast_search`: Search for relevant podcasts

## Task Management

Monitor and control tasks with these endpoints:

- `/list_tasks`: View all running automation tasks
- `/task_status/{task_id}`: Check status of a specific task
- `/stop_task/{task_id}`: Stop a running task

## AI Usage Analytics

### Web Interface

Access the built-in reporting dashboard:
```
GET /ai-usage
```

Query parameters:
- `start_date`: Filter from this date (YYYY-MM-DD)
- `end_date`: Filter to this date (YYYY-MM-DD)
- `group_by`: Group results by "model", "workflow", "endpoint", or "podcast_id"
- `format`: Output format ("json", "text", or "csv")

### Per-Podcast Analytics

View costs for specific podcasts:
```
GET /podcast-cost/{podcast_id}
GET /podcast-cost-dashboard/{podcast_id}
```

### Command-Line Reporting

Generate detailed reports:
```
python generate_ai_usage_report.py --format json --output report.json
```

For comprehensive AI usage tracking documentation, see [AI_USAGE_TRACKING.md](AI_USAGE_TRACKING.md).

## Running the Application

Start the FastAPI server:
```
uvicorn main_fastapi:app --host 0.0.0.0 --port 8000 --reload
```

The application will be available at: http://localhost:8000

## Authentication

- `/login`: User authentication page
- `/logout`: End user session
- `/admin`: Admin dashboard (requires admin privileges)

## API Documentation

FastAPI automatically generates interactive API documentation:
- Swagger UI: http://localhost:5000/docs
- ReDoc: http://localhost:5000/redoc
