# src/config.py

# --- Report Configuration ---
WEEKS_TO_REPORT = 5 # Current week + previous 4 weeks

# Statuses from Airtable needed for the report calculations
TRACKED_STATUSES = [
    "Instantly",
    "Outreached",
    "Responded",
    "Interested",
    "Pending Intro Call Booking",
    "Lost",
    "Form Submitted",
    "Intro Call Booked",
    "Pending Podcast Booking",
    "Recording Booked",
    "Recorded",
    "Live Link",
    "Bounced",
    "Paid"
]

# Statuses to ignore when calculating "Total Records This Week"
STATUSES_TO_IGNORE_IN_TOTAL = [
    "Prospect",
    "OR Ready",
    "Fit",
    "Not a fit",
    "Episode and angles selected",
    "Pitch Done"
]

# Define the order and mapping for rows in the sheet and Slack report
REPORT_ROW_ORDER = [
    {"display_name": "Sent to Instantly", "source_status": "Instantly"},
    {"display_name": "Messages sent", "source_status": "Outreached"},
    {"display_name": "Total replies", "source_status": "Responded"},
    {"display_name": "Positive replies", "source_status": "Interested"},
    {"display_name": "Form Submitted", "source_status": "Form Submitted"},
    {"display_name": "Meetings booked", "source_status": "Pending Intro Call Booking"},
    {"display_name": "Lost", "source_status": "Lost"},
]

# --- Client Exclusions from Reporting ---
EXCLUDED_CLIENTS_FROM_REPORT = [
    "Daniel Borba",
    "Brandon C. White",
    "Kevin Bibelhausen",
    "Cody Schneider",
    "Michael Greenberg"
]

# ... (rest of config.py) ...
print(f"DEBUG: REPORT_ROW_ORDER in config.py: {REPORT_ROW_ORDER}")