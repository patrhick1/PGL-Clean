# PGL Automation Project

This project contains automation workflows for podcast guest management and outreach.

## Features

- Generate bios and angles for podcast guests
- Search for relevant podcasts 
- Determine potential fit of guests for podcasts
- Automate pitch creation and sending
- Track AI token usage and costs

## Getting Started

1. Create a virtual environment:
   ```
   python -m venv pgl_env
   ```

2. Activate the virtual environment:
   - Windows: `.\pgl_env\Scripts\activate`
   - Mac/Linux: `source pgl_env/bin/activate`

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys:
   ```
   ANTHROPIC_API=your_anthropic_api_key
   OPENAI_API=your_openai_api_key
   ```

## Automation Workflows

The main FastAPI application (`main_fastapi.py`) exposes several endpoints to trigger different automation workflows:

- `/trigger_automation?action=summary_host_guest`: Process podcast summaries and identify hosts and guests
- `/trigger_automation?action=enrich_host_name`: Enrich host name data
- `/trigger_automation?action=determine_fit`: Determine if a guest is a good fit for a podcast
- `/trigger_automation?action=pitch_episode_angle`: Select episodes and angles for pitches
- `/trigger_automation?action=pitch_writer`: Generate pitch content
- `/trigger_automation?action=send_pitch`: Send pitches via email

## AI Usage Tracking

This project includes a comprehensive AI usage tracking system that monitors and logs all AI model interactions. Key features include:

- Token usage tracking (input and output)
- Cost calculation based on current API pricing
- Request timing metrics
- Usage reporting by model, workflow, or endpoint

### Viewing AI Usage Reports

#### Web Interface

Access the built-in reporting interface at:
```
GET /ai-usage
```

Optional parameters:
- `start_date`: Filter from this date (ISO format: YYYY-MM-DD)
- `end_date`: Filter to this date (ISO format: YYYY-MM-DD)
- `group_by`: Group results by "model", "workflow", or "endpoint"

#### Command-Line Tool

Generate detailed reports using the command-line tool:
```
python generate_ai_usage_report.py --format json --output monthly_report.json
```

For more details, see [AI_USAGE_TRACKING.md](AI_USAGE_TRACKING.md).

## Running the Application

Start the FastAPI server:
```
python main_fastapi.py
```

The API will be available at: http://localhost:5000
