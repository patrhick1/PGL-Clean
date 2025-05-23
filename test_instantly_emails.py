import os
import pprint
from dotenv import load_dotenv

# Attempt to import InstantlyAPI from the src directory
try:
    from src.external_api_service import InstantlyAPI
except ImportError as e:
    print(f"Error: Could not import InstantlyAPI: {e}")
    print("Please ensure src/external_api_service.py exists and the script is run from the workspace root, or adjust import paths.")
    exit(1)

# Load environment variables from .env file
load_dotenv()

def main():
    """
    Fetches and prints the first 5 emails from Instantly to inspect the data structure.
    """
    print("--- Testing InstantlyAPI: list_emails() ---")

    # Ensure the API key is available
    api_key = os.getenv("INSTANTLY_API_KEY")
    if not api_key:
        print("Error: INSTANTLY_API_KEY not found in environment variables.")
        print("Please ensure your .env file is correctly set up with INSTANTLY_API_KEY.")
        return

    try:
        instantly_client = InstantlyAPI()
        print("InstantlyAPI client initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize InstantlyAPI client: {e}")
        return

    print("\nFetching the first 5 emails from Instantly...")
    try:
        # Fetch a small number of emails, e.g., 5
        email_data = instantly_client.list_emails(limit=5)

        if email_data:
            print("\n--- Email Data Received ---")
            #pprint.pprint(email_data)
            
            items = email_data.get('items', [])
            print(f"\nNumber of email items received: {len(items)}")
            if items:
                for item in items:
                    print("\n--- Structure of the email item ---")
                    pprint.pprint(item)
        else:
            print("\nNo data returned from list_emails. This could be due to an API error (check logs from InstantlyAPI class), no emails, or connectivity issues.")

    except Exception as e:
        print(f"\nAn error occurred while calling list_emails: {e}")

if __name__ == "__main__":
    main()
    print("\n--- Test Script Finished ---") 