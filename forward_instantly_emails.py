import requests
import smtplib
from email.mime.text import MIMEText
import os
from dotenv import load_dotenv
from datetime import datetime, timezone # Added for date comparison
import time # Added for potential delays

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Get credentials from environment variables
# Create a .env file in the same directory with these variables:
# INSTANTLY_API_TOKEN=YOUR_TOKEN_HERE
# GMAIL_USER=yourmasterinbox@gmail.com
# GMAIL_APP_PASSWORD=your_gmail_app_password 
# Note: For Gmail, use an App Password if 2-Factor Authentication is enabled.
# How to generate an App Password: https://support.google.com/accounts/answer/185833
API_TOKEN = os.getenv('INSTANTLY_API_KEY')
GMAIL_USER = os.getenv('GMAIL_USER')
GMAIL_PASS = os.getenv('GMAIL_APP_PASSWORD') # Use App Password if 2FA is enabled

BASE_URL = 'https://api.instantly.ai/api/v2/emails'

# --- Target Date for Filtering ---
TARGET_DATE_STR = "2025-01-01T00:00:00Z" # Start date (inclusive)
TARGET_DATE = datetime.fromisoformat(TARGET_DATE_STR.replace('Z', '+00:00'))

# --- Error Handling for Missing Configuration ---
if not API_TOKEN:
    raise ValueError("INSTANTLY_API_KEY not found in environment variables. Please set it in your .env file.")
if not GMAIL_USER:
    raise ValueError("GMAIL_USER not found in environment variables. Please set it in your .env file.")
if not GMAIL_PASS:
    raise ValueError("GMAIL_APP_PASSWORD not found in environment variables. Please set it in your .env file.")

# --- Configuration & Constants ---
SENT_IDS_FILE = 'sent_email_ids.txt' # File to store IDs of successfully forwarded emails
INITIAL_SENT_COUNT = 2000 # Number of emails assumed sent in the first run

# --- 1. Fetch All Emails from Instantly Unibox ---
filtered_emails = []
starting_after = None
fetched_count = 0 # Total emails checked across all batches
filtered_found_count = 0 # Total emails matching criteria

print(f"Starting to fetch emails from Instantly Unibox created on or after {TARGET_DATE_STR}...")

while True:
    params = {
        # Use normal limit
        "limit": 100, 
    }
    if starting_after:
        params['starting_after'] = starting_after

    headers = {
        "Authorization": f"Bearer {API_TOKEN}"
    }

    try:
        response = requests.get(BASE_URL, headers=headers, params=params)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()

        emails_in_batch = data.get('items', [])
        if not emails_in_batch:
            print("No more emails found in this batch.")
            break # Exit loop if no items are returned

        # Increment total fetched count
        batch_fetched = len(emails_in_batch)
        fetched_count += batch_fetched

        # Filter emails in the current batch
        batch_filtered_count = 0
        for email in emails_in_batch:
            timestamp_str = email.get('timestamp_created')
            if timestamp_str:
                try:
                    # Parse the timestamp string (assuming ISO 8601 format with Z)
                    email_date = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    # Compare with target date
                    if email_date >= TARGET_DATE:
                        filtered_emails.append(email)
                        batch_filtered_count += 1
                except ValueError:
                    print(f"Warning: Could not parse timestamp '{timestamp_str}' for email ID {email.get('id')}")
            else:
                 print(f"Warning: Missing 'timestamp_created' for email ID {email.get('id')}")

        # Update overall filtered count
        filtered_found_count += batch_filtered_count
        print(f"Checked {batch_fetched} emails in this batch. Found {batch_filtered_count} matching criteria. Total matching so far: {filtered_found_count}")

        # Pagination: Get the ID for the next page
        starting_after = data.get('next_starting_after')
        if not starting_after:
            print("Reached the end of emails.")
            break # Exit loop if there's no next page marker

    except requests.exceptions.RequestException as e:
        print(f"Error fetching emails from Instantly API: {e}")
        if response is not None:
            print(f"Response Status Code: {response.status_code}")
            print(f"Response Text: {response.text}")
        break # Exit loop on error
    except Exception as e:
        print(f"An unexpected error occurred during fetching: {e}")
        break # Exit loop on unexpected error


print(f"\nFinished fetching. Checked a total of {fetched_count} emails.")
print(f"Found {len(filtered_emails)} emails created on or after {TARGET_DATE_STR}.")

# --- Pre-populate sent IDs file if it doesn't exist (based on initial run) ---
if not os.path.exists(SENT_IDS_FILE) and len(filtered_emails) >= INITIAL_SENT_COUNT:
    print(f"\n{SENT_IDS_FILE} not found. Populating with the first {INITIAL_SENT_COUNT} assumed sent email IDs...")
    try:
        with open(SENT_IDS_FILE, 'w') as f:
            for i in range(INITIAL_SENT_COUNT):
                email_id = filtered_emails[i].get('id')
                if email_id:
                    f.write(email_id + '\n')
        print(f"Successfully wrote {INITIAL_SENT_COUNT} IDs to {SENT_IDS_FILE}.")
    except IOError as e:
        print(f"Error writing initial IDs to {SENT_IDS_FILE}: {e}. Please check permissions.")
        # Decide if we should exit or continue without persistence
        # For safety, let's exit if we can't write the initial state
        exit(1)
elif not os.path.exists(SENT_IDS_FILE):
     print(f"\nWarning: {SENT_IDS_FILE} not found, and fewer than {INITIAL_SENT_COUNT} emails were filtered. Starting with an empty sent list.")
     # Create the file so it exists for appending later
     try:
         open(SENT_IDS_FILE, 'w').close()
     except IOError as e:
         print(f"Error creating empty {SENT_IDS_FILE}: {e}. Please check permissions. Exiting.")
         exit(1)

# --- Function to load sent IDs ---
def load_sent_ids(filename):
    sent_ids = set()
    try:
        with open(filename, 'r') as f:
            for line in f:
                sent_ids.add(line.strip())
    except FileNotFoundError:
        print(f"Info: {filename} not found. Starting with an empty list of sent emails.")
    except IOError as e:
        print(f"Error reading {filename}: {e}. Proceeding without previously sent IDs.")
    return sent_ids

# --- Function to save a sent ID ---
def save_sent_id(filename, email_id):
    try:
        with open(filename, 'a') as f:
            f.write(email_id + '\n')
    except IOError as e:
        print(f"Error writing ID {email_id} to {filename}: {e}")


# --- 2. Send/Forward Emails to Master Gmail Inbox via SMTP ---
if filtered_emails:
    # Load IDs that have already been sent
    sent_ids = load_sent_ids(SENT_IDS_FILE)
    print(f"\nLoaded {len(sent_ids)} previously sent email IDs from {SENT_IDS_FILE}.")

    emails_to_forward = [email for email in filtered_emails if email.get('id') not in sent_ids]
    total_to_forward = len(emails_to_forward)
    print(f"Attempting to forward {total_to_forward} remaining emails...")

    if total_to_forward == 0:
        print("No new emails to forward.")
    else:
        print("\nStarting to forward emails to Gmail...")
        forwarded_count = 0 # Count for this run
        failed_count = 0    # Count for this run
        smtp_server = None  # Initialize smtp_server
        try:
            # Connect to Gmail SMTP server
            smtp_server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            smtp_server.ehlo() # Identify ourselves to the SMTP server
            smtp_server.login(GMAIL_USER, GMAIL_PASS)
            print("Successfully connected to Gmail SMTP server.")

            # Iterate only through emails that haven't been sent yet
            for email in emails_to_forward:
                email_id = email.get('id')
                subject = email.get('subject', 'No Subject') # Get subject early for logging

                # Skip if ID is missing for some reason
                if not email_id:
                    print(f"Warning: Skipping email with missing ID.")
                    continue

                # Double-check just in case (should be filtered by list comprehension above)
                if email_id in sent_ids:
                    print(f"Skipping email ID {email_id} (already marked as sent).")
                    continue

                try:
                    # Construct the email message
                    # Use HTML body if available, otherwise fallback to text
                    body = email.get('body', {}).get('html') or email.get('body', {}).get('text', 'No body content')
                    # Ensure body is a string
                    if not isinstance(body, str):
                        body = str(body)

                    msg = MIMEText(body, 'html' if email.get('body', {}).get('html') else 'plain', 'utf-8') # Specify UTF-8

                    msg['Subject'] = f"FWD via Instantly: {subject}"
                    msg['From'] = GMAIL_USER
                    msg['To'] = GMAIL_USER

                    original_sender = email.get('from_address_email')
                    if original_sender:
                        msg['Reply-To'] = original_sender
                        msg.add_header('X-Original-Sender', original_sender)

                    original_recipients = ", ".join(email.get('to_address_email_list', []))
                    if original_recipients:
                        msg.add_header('X-Original-Recipients', original_recipients)

                    original_timestamp = email.get('timestamp_email')
                    if original_timestamp:
                         msg.add_header('X-Original-Timestamp', str(original_timestamp))

                    # Send the email
                    smtp_server.sendmail(GMAIL_USER, GMAIL_USER, msg.as_string())
                    
                    # If successful, save the ID and add to current set
                    save_sent_id(SENT_IDS_FILE, email_id)
                    sent_ids.add(email_id) # Add to set to prevent re-sending in this run if logic error
                    forwarded_count += 1
                    if forwarded_count % 50 == 0: # Print progress every 50 emails
                         # Show progress relative to this run's target
                         print(f"Forwarded {forwarded_count}/{total_to_forward} emails this run...")

                except smtplib.SMTPException as e:
                    print(f"SMTP error sending email ID {email_id} with subject '{subject}': {e}")
                    failed_count += 1
                    # If the error is likely a rate limit, stop this run
                    # (Exact error codes/messages for rate limits can vary)
                    if isinstance(e, smtplib.SMTPSenderRefused) or isinstance(e, smtplib.SMTPRecipientsRefused) or "limit" in str(e).lower():
                         print("Stopping run due to potential rate limit or refusal.")
                         break # Exit the loop
                    # Optional: Add a small delay on other SMTP errors
                    # time.sleep(5)
                except Exception as e:
                    print(f"Error processing or sending email ID {email_id} with subject '{subject}': {e}")
                    failed_count += 1

            print("\nFinished forwarding attempt for this run.")
            print(f"Successfully forwarded in this run: {forwarded_count}")
            print(f"Failed to forward in this run: {failed_count}")

        except smtplib.SMTPAuthenticationError:
            print("\nSMTP Authentication Error: Failed to login to Gmail. Check your GMAIL_USER and GMAIL_APP_PASSWORD.")
        except smtplib.SMTPException as e:
            print(f"\nSMTP Connection Error: {e}")
        except Exception as e:
            print(f"\nAn unexpected error occurred during SMTP forwarding: {e}")
        finally:
            # Ensure server connection is closed
            if smtp_server:
                try:
                    smtp_server.quit()
                    print("SMTP server connection closed.")
                except smtplib.SMTPException:
                    pass # Ignore errors during quit
else:
    print("\nNo emails matching criteria were found, so no forwarding needed.")

print("\nScript finished.") 