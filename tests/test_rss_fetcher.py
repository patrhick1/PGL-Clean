# test_rss_fetcher.py
import logging
import pprint
import sys
import os

# Ensure the main script's directory is in the path to import necessary modules
# Adjust the path if your test file is located differently relative to the main scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming your main scripts (like fetch_episodes, airtable_service) are in the same directory
sys.path.insert(0, script_dir) # If tests are in the same dir as scripts

# Import the specific function to test
try:
    from fetch_episodes import fetch_episodes_from_rss
except ImportError as e:
    print(f"Error importing fetch_episodes_from_rss: {e}")
    print("Please ensure test_rss_fetcher.py is in the same directory as fetch_episodes.py or adjust sys.path.")
    sys.exit(1)

# Configure basic logging to see output from the function being tested
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

# --- Configuration ---
# The specific RSS feed URL to test (provided by user)
TEST_RSS_URL = "https://feeds.buzzsprout.com/1300039.rss"
MAX_EPISODES_TO_FETCH = 10 # Or set to a different number if needed for testing
# ---

def run_rss_fetch_test():
    """Runs the test for fetch_episodes_from_rss."""
    print(f"--- Testing fetch_episodes_from_rss ---")
    print(f"Feed URL: {TEST_RSS_URL}")
    print(f"Max Episodes to Fetch: {MAX_EPISODES_TO_FETCH}")
    print("-" * 40)

    try:
        # Call the function directly
        # Pass stop_flag=None as we aren't testing threading here
        extracted_episodes = fetch_episodes_from_rss(
            rss_url=TEST_RSS_URL,
            max_episodes=MAX_EPISODES_TO_FETCH,
            stop_flag=None
        )

        print(f"\n--- Results ({len(extracted_episodes)} episodes extracted) ---")
        if extracted_episodes:
            # Pretty-print the list of episode dictionaries
            pprint.pprint(extracted_episodes)
            print("\n--- Key fields extracted (first episode example) ---")
            if extracted_episodes:
                first_ep = extracted_episodes[0]
                print(f"  Title: {first_ep.get('episode_title')}")
                print(f"  Description (start): {first_ep.get('episode_description', '')[:100]}...")
                print(f"  GUID/ID: {first_ep.get('episode_id')}")
                print(f"  Audio URL: {first_ep.get('episode_audio_url')}")
                print(f"  Posted At (datetime): {first_ep.get('posted_at')}")
                print(f"  Web Link: {first_ep.get('episode_url')}")
        else:
            print("No episodes were successfully extracted from the feed.")
            print("(Check logs above for any parsing warnings or errors)")

        print("\n--- Test Finished ---")

    except Exception as e:
        logging.error(f"Test script encountered an unexpected error: {e}", exc_info=True)
        print("\n--- Test Failed due to script error ---")

if __name__ == "__main__":
    run_rss_fetch_test() 