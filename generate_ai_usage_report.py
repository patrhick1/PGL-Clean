#!/usr/bin/env python
"""
AI Usage Report Generator

This script generates reports on AI usage from the stored logs.
It can output reports in different formats (text, CSV, JSON) and filter by date range.

Example usage:
  python generate_ai_usage_report.py --format json
  python generate_ai_usage_report.py --start-date 2024-05-01 --end-date 2024-05-31 --group-by workflow
  python generate_ai_usage_report.py --format csv --output usage_report.csv
  python generate_ai_usage_report.py --podcast-id recXXXXXXXXXXXXX --format json

Author: Generated for PGL Project
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path
from tabulate import tabulate
from typing import Optional, Dict, Any, List

# Import the tracker
from ai_usage_tracker import tracker


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate AI usage reports")
    
    parser.add_argument(
        "--start-date", 
        type=str,
        help="Start date for the report (ISO format: YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end-date", 
        type=str,
        help="End date for the report (ISO format: YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--group-by", 
        type=str,
        choices=["model", "workflow", "endpoint", "podcast_id"],
        default="model",
        help="Group results by this field"
    )
    
    parser.add_argument(
        "--format", 
        type=str,
        choices=["text", "json", "csv"],
        default="text",
        help="Output format for the report"
    )
    
    parser.add_argument(
        "--output", 
        type=str,
        help="File to write the report to (default: stdout)"
    )
    
    parser.add_argument(
        "--podcast-id",
        type=str,
        help="Airtable record ID to generate a podcast-specific report"
    )
    
    return parser.parse_args()


def format_as_text(report: Dict[str, Any]) -> str:
    """Format the report as human-readable text."""
    text_output = []
    
    # Header
    text_output.append("=" * 60)
    text_output.append("AI USAGE REPORT")
    text_output.append("=" * 60)
    
    # Special handling for podcast-specific reports
    if "podcast_id" in report:
        text_output.append(f"Podcast ID: {report['podcast_id']}")
        text_output.append("-" * 60)
        text_output.append(f"Total API calls: {report['total_calls']}")
        text_output.append(f"Total tokens: {report['total_tokens']:,}")
        text_output.append(f"Total cost: ${report['total_cost']:.4f}")
        text_output.append("-" * 60)
        
        # Workflow stages breakdown
        text_output.append("\nBreakdown by workflow stage:")
        text_output.append("-" * 60)
        
        # Create a table for the workflow stages
        table_data = []
        for stage_name, stats in report['workflow_stages'].items():
            table_data.append([
                stage_name, 
                stats['calls'],
                f"{stats['tokens']:,}",
                f"${stats['cost']:.4f}"
            ])
        
        headers = ["Workflow Stage", "Calls", "Tokens", "Cost"]
        text_output.append(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Timeline of operations
        text_output.append("\nTimeline of operations:")
        text_output.append("-" * 60)
        
        timeline_data = []
        for entry in report['timeline']:
            timeline_data.append([
                entry['timestamp'],
                entry['workflow'],
                entry['model'],
                f"{entry['tokens']:,}",
                f"${entry['cost']:.4f}"
            ])
        
        timeline_headers = ["Timestamp", "Workflow", "Model", "Tokens", "Cost"]
        text_output.append(tabulate(timeline_data, headers=timeline_headers, tablefmt="grid"))
        
        return "\n".join(text_output)
    
    # Standard report format
    date_range = f"Date range: "
    if report.get("start_date"):
        date_range += f"From {report['start_date']} "
    if report.get("end_date"):
        date_range += f"To {report['end_date']} "
    if not report.get("start_date") and not report.get("end_date"):
        date_range += "All time"
    text_output.append(date_range)
    
    # Summary stats
    text_output.append("-" * 60)
    text_output.append(f"Total API calls: {report['total_entries']}")
    text_output.append(f"Total tokens: {report['total_tokens']:,}")
    text_output.append(f"Total cost: ${report['total_cost']:.2f}")
    text_output.append("-" * 60)
    
    # Group details
    text_output.append(f"\nBreakdown by {report['grouped_by']}:")
    text_output.append("-" * 60)
    
    # Create a table for the groups
    table_data = []
    for group_name, stats in report['groups'].items():
        table_data.append([
            group_name, 
            stats['calls'],
            f"{stats['tokens_in']:,}",
            f"{stats['tokens_out']:,}",
            f"{stats['total_tokens']:,}",
            f"${stats['cost']:.2f}",
            f"{stats['avg_time']:.2f} sec"
        ])
    
    headers = ["Name", "Calls", "Input Tokens", "Output Tokens", "Total Tokens", "Cost", "Avg Time"]
    text_output.append(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    return "\n".join(text_output)


def format_as_csv(report: Dict[str, Any]) -> str:
    """Format the report as CSV."""
    csv_output = []
    
    # Special handling for podcast-specific reports
    if "podcast_id" in report:
        # Header row for podcast report
        csv_output.append("Podcast ID,Workflow Stage,Calls,Tokens,Cost")
        
        # Data rows for workflow stages
        for stage_name, stats in report['workflow_stages'].items():
            csv_output.append(
                f"{report['podcast_id']},{stage_name},{stats['calls']}," +
                f"{stats['tokens']},{stats['cost']:.6f}"
            )
        
        # Add a summary row
        csv_output.append(f"{report['podcast_id']},TOTAL,{report['total_calls']}," +
                         f"{report['total_tokens']},{report['total_cost']:.6f}")
        
        return "\n".join(csv_output)
    
    # Standard report format
    csv_output.append(f"{report['grouped_by']},Calls,Input Tokens,Output Tokens,Total Tokens,Cost,Avg Time (sec)")
    
    # Data rows
    for group_name, stats in report['groups'].items():
        csv_output.append(
            f"{group_name},{stats['calls']},{stats['tokens_in']},{stats['tokens_out']}," +
            f"{stats['total_tokens']},{stats['cost']:.6f},{stats['avg_time']:.3f}"
        )
    
    # Add a summary row
    csv_output.append(f"TOTAL,{report['total_entries']},,," +
                     f"{report['total_tokens']},{report['total_cost']:.6f},")
    
    return "\n".join(csv_output)


def main():
    """Main function to generate the report."""
    args = parse_arguments()
    
    # Check if we should generate a podcast-specific report
    if args.podcast_id:
        report = tracker.get_record_cost_report(args.podcast_id)
    else:
        # Generate standard report
        report = tracker.generate_report(
            start_date=args.start_date,
            end_date=args.end_date,
            group_by=args.group_by
        )
    
    # Handle error in report generation
    if "error" in report:
        print(f"Error: {report['error']}", file=sys.stderr)
        sys.exit(1)
    
    # Format the report based on requested format
    if args.format == "text":
        output = format_as_text(report)
    elif args.format == "json":
        output = json.dumps(report, indent=2)
    elif args.format == "csv":
        output = format_as_csv(report)
    else:
        print(f"Error: Unsupported format {args.format}", file=sys.stderr)
        sys.exit(1)
    
    # Output the report
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Report written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main() 