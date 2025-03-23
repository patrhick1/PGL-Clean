# AI Usage Tracking System

This document explains how the AI usage tracking system works within the PGL application, its features, and how to use the reporting tools.

## Overview

The AI usage tracking system monitors and logs all interactions with AI models (OpenAI and Anthropic) throughout the application. It tracks:

- **Input and output tokens**: Measures the size of prompts and responses
- **Cost calculation**: Estimates the cost of each API call based on current pricing
- **Timing**: Measures execution time for each request
- **Context identification**: Associates requests with specific workflows and endpoints
- **Model identification**: Records which AI model was used
- **Podcast-level tracking**: Associates AI usage with specific Airtable podcast records

## How It Works

The tracker has been integrated with both OpenAI and Anthropic service classes, automatically capturing usage data whenever these services are called. The data is logged to a CSV file (`ai_usage_logs.csv`) for future analysis.

### Key Components

1. **AI Usage Tracker** (`ai_usage_tracker.py`): Core module that logs AI usage data
2. **Service Integration**: Hooks in `anthropic_service.py` and `openai_service.py` that call the tracker
3. **Reporting Endpoints**: FastAPI endpoints at `/ai-usage` and `/podcast-cost/{podcast_id}` for viewing reports
4. **Command-Line Tool**: Script for generating detailed reports (`generate_ai_usage_report.py`)

## Viewing Usage Reports

### Web Interface

Access the built-in reporting interface at:

```
GET /ai-usage
```

Optional query parameters:
- `start_date`: Filter from this date (ISO format: YYYY-MM-DD)
- `end_date`: Filter to this date (ISO format: YYYY-MM-DD)
- `group_by`: Group results by "model", "workflow", "endpoint", or "podcast_id" (default: "model")

Example:
```
GET /ai-usage?start_date=2024-05-01&end_date=2024-05-31&group_by=workflow
```

### Podcast-Specific Cost Reports

To get detailed cost reports for a specific podcast:

```
GET /podcast-cost/{podcast_id}
```

Where `{podcast_id}` is the Airtable podcast ID.

This endpoint provides a complete breakdown of AI usage and costs for processing a specific podcast through the entire pipeline.

### Command-Line Tool

For more detailed reports or scheduled analysis, use the command-line tool:

```bash
# View overall usage statistics
python generate_ai_usage_report.py

# Filter by date range
python generate_ai_usage_report.py --start-date 2024-05-01 --end-date 2024-05-31

# Group by different fields
python generate_ai_usage_report.py --group-by workflow
python generate_ai_usage_report.py --group-by endpoint
python generate_ai_usage_report.py --group-by podcast_id

# Output in different formats
python generate_ai_usage_report.py --format json
python generate_ai_usage_report.py --format csv

# Save to file
python generate_ai_usage_report.py --output monthly_report.txt
python generate_ai_usage_report.py --format csv --output usage_data.csv

# Generate report for a specific podcast (by Airtable podcast ID)
python generate_ai_usage_report.py --podcast-id recXXXXXXXXXXXXX --format json
```

## Sample Report Output

### Text Format

```
============================================================
AI USAGE REPORT
============================================================
Date range: From 2024-05-01 To 2024-05-31
------------------------------------------------------------
Total API calls: 1,250
Total tokens: 2,456,789
Total cost: $45.23
------------------------------------------------------------

Breakdown by model:
------------------------------------------------------------
+---------------------------+-------+---------------+----------------+---------------+--------+-----------+
| Name                      | Calls | Input Tokens  | Output Tokens  | Total Tokens  | Cost   | Avg Time  |
+===========================+=======+===============+================+===============+========+===========+
| gpt-4o-2024-08-06         | 870   | 1,234,567     | 345,678        | 1,580,245     | $36.45 | 1.25 sec  |
+---------------------------+-------+---------------+----------------+---------------+--------+-----------+
| claude-3-5-haiku-20241022 | 380   | 678,900       | 197,644        | 876,544       | $8.78  | 0.92 sec  |
+---------------------------+-------+---------------+----------------+---------------+--------+-----------+
```

### Podcast-Specific Report Format

```
============================================================
AI USAGE REPORT
============================================================
Podcast ID: recXXXXXXXXXXXXX
------------------------------------------------------------
Total API calls: 42
Total tokens: 156,789
Total cost: $1.23
------------------------------------------------------------

Breakdown by workflow stage:
------------------------------------------------------------
+------------------+-------+----------+--------+
| Workflow Stage   | Calls | Tokens   | Cost   |
+==================+=======+==========+========+
| discovery        | 12    | 45,678   | $0.32  |
+------------------+-------+----------+--------+
| content_analysis | 15    | 67,890   | $0.56  |
+------------------+-------+----------+--------+
| email_creation   | 15    | 43,221   | $0.35  |
+------------------+-------+----------+--------+

Timeline of operations:
------------------------------------------------------------
+---------------------+------------------+---------------------------+----------+--------+
| Timestamp           | Workflow         | Model                     | Tokens   | Cost   |
+=====================+==================+===========================+==========+========+
| 2024-05-01T09:23:45 | discovery        | gpt-4o-2024-08-06         | 12,345   | $0.08  |
+---------------------+------------------+---------------------------+----------+--------+
| 2024-05-01T09:25:12 | discovery        | claude-3-5-haiku-20241022 | 8,765    | $0.02  |
+---------------------+------------------+---------------------------+----------+--------+
| ...                 | ...              | ...                       | ...      | ...    |
+---------------------+------------------+---------------------------+----------+--------+
```

### JSON Format

```json
{
  "start_date": "2024-05-01",
  "end_date": "2024-05-31",
  "total_entries": 1250,
  "total_tokens": 2456789,
  "total_cost": 45.23,
  "grouped_by": "model",
  "groups": {
    "gpt-4o-2024-08-06": {
      "calls": 870,
      "tokens_in": 1234567,
      "tokens_out": 345678,
      "total_tokens": 1580245,
      "cost": 36.45,
      "avg_time": 1.25
    },
    "claude-3-5-haiku-20241022": {
      "calls": 380,
      "tokens_in": 678900,
      "tokens_out": 197644,
      "total_tokens": 876544,
      "cost": 8.78,
      "avg_time": 0.92
    }
  }
}
```

### Podcast-Specific JSON Format

```json
{
  "podcast_id": "recXXXXXXXXXXXXX",
  "total_cost": 1.23,
  "total_tokens": 156789,
  "total_calls": 42,
  "workflow_stages": {
    "discovery": {
      "calls": 12,
      "tokens": 45678,
      "cost": 0.32
    },
    "content_analysis": {
      "calls": 15,
      "tokens": 67890,
      "cost": 0.56
    },
    "email_creation": {
      "calls": 15,
      "tokens": 43221,
      "cost": 0.35
    }
  },
  "timeline": [
    {
      "timestamp": "2024-05-01T09:23:45",
      "workflow": "discovery",
      "model": "gpt-4o-2024-08-06",
      "tokens": 12345,
      "cost": 0.08
    },
    {
      "timestamp": "2024-05-01T09:25:12",
      "workflow": "discovery",
      "model": "claude-3-5-haiku-20241022",
      "tokens": 8765,
      "cost": 0.02
    }
  ]
}
```

## Cost Calculation

The system uses the following cost rates (as of May 2024):

### OpenAI
- **gpt-4o-2024-08-06**: $0.005 per 1K input tokens, $0.015 per 1K output tokens

### Anthropic
- **claude-3-5-haiku-20241022**: $0.00025 per 1K input tokens, $0.00125 per 1K output tokens

These rates can be updated in the `COST_RATES` dictionary in `ai_usage_tracker.py` if pricing changes.

## Adding Tracking to New AI Services

If you add new AI services or want to track usage in additional places, follow these steps:

1. Import the tracker:
```python
from ai_usage_tracker import tracker
```

2. Add timing code around your API call:
```python
import time
start_time = time.time()
response = ai_client.some_api_call(...)
execution_time = time.time() - start_time
```

3. Log the usage:
```python
tracker.log_usage(
    workflow="your_workflow_name",
    model="model_name",
    tokens_in=response.usage.prompt_tokens,  # Adjust based on the API response structure
    tokens_out=response.usage.completion_tokens,
    execution_time=execution_time,
    endpoint="service.api.endpoint",
    podcast_id=airtable_podcast_id  # Optional: Include Airtable podcast ID for tracking
)
```

## Tracking Podcast-Specific Costs

To track AI usage associated with specific podcasts:

1. When calling AI services, pass the Airtable podcast ID:

```python
# Example with Anthropic service
response = anthropic_service.create_message(
    prompt="Your prompt here",
    workflow="podcast_analysis",
    podcast_id="recXXXXXXXXXXXXX"  # Airtable podcast ID
)

# Example with OpenAI service
data = openai_service.transform_text_to_structured_data(
    prompt="Your prompt",
    raw_text="Text to analyze",
    data_type="Structured",
    workflow="extract_podcast_topics",
    podcast_id="recXXXXXXXXXXXXX"  # Airtable podcast ID
)
```

2. Generate podcast-specific reports using the API or command line tool:

```bash
# API endpoint
GET /podcast-cost/recXXXXXXXXXXXXX

# Command line
python generate_ai_usage_report.py --podcast-id recXXXXXXXXXXXXX --format json
```

## Troubleshooting

- **Missing logs**: Ensure the application has write permissions for the `ai_usage_logs.csv` file
- **Incorrect costs**: Check the `COST_RATES` dictionary in `ai_usage_tracker.py` to ensure rates are current
- **No data in reports**: Verify that the relevant service integrations are correctly calling the tracker 
- **Missing podcast ID data**: Ensure Airtable podcast IDs are being passed to the AI service methods