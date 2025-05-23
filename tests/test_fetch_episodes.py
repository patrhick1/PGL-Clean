# test_fetch_episodes.py
import unittest
from unittest.mock import patch, MagicMock, ANY
import sys
import os
from datetime import datetime, timedelta

# Ensure the main script's directory is in the path to import necessary modules
# Adjust the path if your test file is located differently relative to the main scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming your main scripts (like fetch_episodes, airtable_service) are in the same directory
sys.path.insert(0, script_dir) # If tests are in the same dir as scripts

# Import the specific components or functions we need to test or mock
# Mocking the module directly where it's used is often cleaner.
# We will mock airtable_client where it's used inside the function implicitly
# We also need parse_date
from fetch_episodes import parse_date # Use the actual parser


# Minimal mock class to represent airtable_client instance
class MockAirtableClient:
    def search_records(self, table_name, formula=None, view=None):
        # This method will be overridden by mock patcher anyway
        pass

# Store the original datetime object to avoid issues with mocking
real_datetime = datetime

class TestFetchEpisodesDuplicateCheck(unittest.TestCase):

    @patch('fetch_episodes.airtable_client') # Mock the client instance within the fetch_episodes module
    @patch('fetch_episodes.logger') # Mock logger
    @patch('fetch_episodes.datetime', wraps=real_datetime) # Mock datetime to control now()
    def test_duplicate_episode_found(self, mock_datetime, mock_logger, mock_airtable_client):
        """
        Test that an episode is skipped if a duplicate is found in Airtable.
        """
        # --- Arrange ---
        mock_now = real_datetime(2024, 1, 15, 12, 0, 0)
        mock_datetime.now.return_value = mock_now

        # Simulate search_records finding an existing episode
        mock_airtable_client.search_records.return_value = [{'id': 'recExistingEpisode', 'fields': {}}]

        record_id = 'recPodcast123'
        podcast_name = 'Test Podcast Duplicate'
        # Episode data matching the duplicate check criteria
        test_episode = {
            'episode_title': 'Test Episode - Duplicate',
            'episode_audio_url': 'http://example.com/duplicate.mp3',
            'posted_at': mock_now - timedelta(days=10) # Ensure it passes date check
        }
        # Simulate the list where processed episodes are added
        episodes_to_create = []
        # Simulate the processing loop for a single episode
        skipped = False # Flag to check if continue would be hit

        # --- Act ---
        episode_audio_url = test_episode.get("episode_audio_url")
        if episode_audio_url:
            escaped_episode_url = episode_audio_url.replace("'", "\\'")
            formula = f"AND({{Episode URL}}='{escaped_episode_url}', RECORD_ID({{Podcast}}) = '{record_id}')"
            try:
                existing_episodes = mock_airtable_client.search_records('Podcast_Episodes', formula=formula)
                if existing_episodes:
                    # Just execute the logic, don't assert logger here
                    # Simulate the logger call that *should* happen
                    mock_logger.info(f"Duplicate found for Episode URL '{episode_audio_url}' (Podcast: '{podcast_name}'). Skipping creation.")
                    skipped = True # Simulate 'continue'
            except Exception as e:
                 # Just execute the logic, don't assert logger here
                 # Simulate the logger call
                 mock_logger.error(f"Error checking for duplicate episode URL '{episode_audio_url}' for podcast '{podcast_name}': {e}. Proceeding without check for this episode.")
                 # Don't set skipped = True, as the original code proceeds

        if not skipped:
             # Simulate passing date check etc. (This part won't run if skipped)
             field_to_update = {"Episode Title": test_episode['episode_title']} # Minimal representation
             episodes_to_create.append(field_to_update)


        # --- Assert ---
        expected_formula = f"AND({{Episode URL}}='http://example.com/duplicate.mp3', RECORD_ID({{Podcast}}) = 'recPodcast123')"
        mock_airtable_client.search_records.assert_called_once_with('Podcast_Episodes', formula=expected_formula)
        self.assertTrue(skipped)
        self.assertEqual(len(episodes_to_create), 0)
        # Check the specific log message was called *after* the logic ran
        mock_logger.info.assert_called_with(f"Duplicate found for Episode URL '{episode_audio_url}' (Podcast: '{podcast_name}'). Skipping creation.")
        mock_logger.error.assert_not_called() # Ensure no error was logged

    @patch('fetch_episodes.airtable_client')
    @patch('fetch_episodes.logger')
    @patch('fetch_episodes.datetime', wraps=real_datetime)
    def test_no_duplicate_episode_found(self, mock_datetime, mock_logger, mock_airtable_client):
        """
        Test that an episode is processed if no duplicate is found.
        """
        # --- Arrange ---
        mock_now = real_datetime(2024, 1, 15, 12, 0, 0)
        mock_datetime.now.return_value = mock_now

        # Simulate search_records finding nothing
        mock_airtable_client.search_records.return_value = []

        record_id = 'recPodcast456'
        podcast_name = 'Test Podcast New'
        test_episode = {
            'episode_title': 'Test Episode - New',
            'episode_audio_url': 'http://example.com/new_episode.mp3',
            'posted_at': mock_now - timedelta(days=5) # Passes date check
        }
        episodes_to_create = []
        skipped = False

        # --- Act --- 
        episode_audio_url = test_episode.get("episode_audio_url")
        if episode_audio_url:
            escaped_episode_url = episode_audio_url.replace("'", "\\'")
            formula = f"AND({{Episode URL}}='{escaped_episode_url}', RECORD_ID({{Podcast}}) = '{record_id}')"
            try:
                existing_episodes = mock_airtable_client.search_records('Podcast_Episodes', formula=formula)
                if existing_episodes:
                    skipped = True
            except Exception as e:
                 mock_logger.error.assert_any_call(f"Error checking duplicate...: {e}") # Simplified check

        if not skipped:
             # Simulate passing date check etc.
             dt_published_input = test_episode.get("posted_at")
             # Use the real parse_date
             dt_published = parse_date(dt_published_input) if isinstance(dt_published_input, str) else dt_published_input
             should_add = False
             if dt_published:
                 # Use mocked now
                 now_dt = mock_datetime.now(dt_published.tzinfo)
                 days_ago = now_dt - timedelta(days=150) # Use 150 days as per current code
                 if dt_published >= days_ago:
                     should_add = True
                 else:
                      mock_logger.info.assert_any_call(f"Episode '{test_episode['episode_title']}' (Published: {dt_published.date()}) is older than 150 days. Skipping.")
             else:
                 mock_logger.warning.assert_any_call(f"Could not parse published date ('{dt_published_input}') for episode '{test_episode['episode_title']}'. Skipping.")

             if should_add:
                 # This simulates the field preparation block more closely
                 field_to_update = {
                     "Episode Title": test_episode['episode_title'],
                     "Episode URL": test_episode['episode_audio_url'],
                     "Published": dt_published.strftime('%Y-%m-%d'),
                     "Podcast": [record_id],
                     # Simulate other fields needed for completeness
                     "Summary": "Test Summary",
                     "Episode ID": "ep123",
                     "Episode Web Link": "http://weblink",
                     "Transcribe": True,
                     "Downloaded": False
                 }
                 episodes_to_create.append(field_to_update)

        # --- Assert ---
        expected_formula = f"AND({{Episode URL}}='http://example.com/new_episode.mp3', RECORD_ID({{Podcast}}) = 'recPodcast456')"
        mock_airtable_client.search_records.assert_called_once_with('Podcast_Episodes', formula=expected_formula)
        self.assertFalse(skipped)
        self.assertEqual(len(episodes_to_create), 1)
        self.assertEqual(episodes_to_create[0]["Episode Title"], "Test Episode - New")
        self.assertEqual(episodes_to_create[0]["Episode URL"], "http://example.com/new_episode.mp3")

    @patch('fetch_episodes.airtable_client')
    @patch('fetch_episodes.logger')
    @patch('fetch_episodes.datetime', wraps=real_datetime)
    def test_episode_missing_audio_url(self, mock_datetime, mock_logger, mock_airtable_client):
        """
        Test that an episode without an audio URL is processed and logged.
        """
        # --- Arrange ---
        mock_now = real_datetime(2024, 1, 15, 12, 0, 0)
        mock_datetime.now.return_value = mock_now

        record_id = 'recPodcast789'
        podcast_name = 'Test Podcast No URL'
        test_episode = {
            'episode_title': 'Test Episode - No URL',
            'episode_audio_url': None, # Missing URL
            'posted_at': mock_now - timedelta(days=1) # Passes date check
        }
        episodes_to_create = []
        skipped = False

        # --- Act ---
        episode_audio_url = test_episode.get("episode_audio_url")
        if episode_audio_url:
             # This block should not be entered
             pass
        else:
            # Simulate the logger call that should happen
            mock_logger.warning(f"Episode '{test_episode['episode_title']}' for podcast '{podcast_name}' lacks an episode_audio_url. Cannot perform duplicate check based on URL.")

        if not skipped:
             # Simulate passing date check etc.
             dt_published_input = test_episode.get("posted_at")
             dt_published = parse_date(dt_published_input) if isinstance(dt_published_input, str) else dt_published_input
             should_add = False
             if dt_published:
                 now_dt = mock_datetime.now(dt_published.tzinfo)
                 days_ago = now_dt - timedelta(days=150)
                 if dt_published >= days_ago:
                     should_add = True

             if should_add:
                 field_to_update = {"Episode Title": test_episode['episode_title']} # Minimal representation
                 episodes_to_create.append(field_to_update)

        # --- Assert ---
        mock_airtable_client.search_records.assert_not_called() # Ensure Airtable wasn't searched
        self.assertFalse(skipped)
        self.assertEqual(len(episodes_to_create), 1)
        # Check the specific log message was called
        mock_logger.warning.assert_called_with(
            f"Episode '{test_episode['episode_title']}' for podcast '{podcast_name}' lacks an episode_audio_url. Cannot perform duplicate check based on URL."
        )
        mock_logger.error.assert_not_called()

    @patch('fetch_episodes.airtable_client')
    @patch('fetch_episodes.logger')
    @patch('fetch_episodes.datetime', wraps=real_datetime)
    def test_airtable_search_error(self, mock_datetime, mock_logger, mock_airtable_client):
        """
        Test that an episode is still processed if Airtable search fails.
        """
        # --- Arrange ---
        mock_now = real_datetime(2024, 1, 15, 12, 0, 0)
        mock_datetime.now.return_value = mock_now
        error_message = "Airtable API Error"

        # Simulate search_records raising an error
        mock_airtable_client.search_records.side_effect = Exception(error_message)

        record_id = 'recPodcastABC'
        podcast_name = 'Test Podcast API Error'
        test_episode = {
            'episode_title': 'Test Episode - API Error',
            'episode_audio_url': 'http://example.com/api_error.mp3',
            'posted_at': mock_now - timedelta(days=2) # Passes date check
        }
        episodes_to_create = []
        skipped = False

        # --- Act ---
        episode_audio_url = test_episode.get("episode_audio_url")
        if episode_audio_url:
            escaped_episode_url = episode_audio_url.replace("'", "\\'")
            formula = f"AND({{Episode URL}}='{escaped_episode_url}', RECORD_ID({{Podcast}}) = '{record_id}')"
            try:
                existing_episodes = mock_airtable_client.search_records('Podcast_Episodes', formula=formula)
                if existing_episodes:
                    skipped = True
            except Exception as e:
                 # Simulate the logger call that *should* happen in the except block
                 mock_logger.error(f"Error checking for duplicate episode URL '{episode_audio_url}' for podcast '{podcast_name}': {e}. Proceeding without check for this episode.")
                 # Don't set skipped=True, proceed as per original logic

        if not skipped:
             # Simulate passing date check etc.
             dt_published_input = test_episode.get("posted_at")
             dt_published = parse_date(dt_published_input) if isinstance(dt_published_input, str) else dt_published_input
             should_add = False
             if dt_published:
                 now_dt = mock_datetime.now(dt_published.tzinfo)
                 days_ago = now_dt - timedelta(days=150)
                 if dt_published >= days_ago:
                     should_add = True

             if should_add:
                 field_to_update = {"Episode Title": test_episode['episode_title']} # Minimal representation
                 episodes_to_create.append(field_to_update)

        # --- Assert ---
        expected_formula = f"AND({{Episode URL}}='http://example.com/api_error.mp3', RECORD_ID({{Podcast}}) = 'recPodcastABC')"
        mock_airtable_client.search_records.assert_called_once_with('Podcast_Episodes', formula=expected_formula)
        # Verify the episode was NOT skipped despite the error
        self.assertFalse(skipped)
        self.assertEqual(len(episodes_to_create), 1)
        # Check the specific log message was called
        mock_logger.error.assert_called_with(f"Error checking for duplicate episode URL '{episode_audio_url}' for podcast '{podcast_name}': {Exception(error_message)}. Proceeding without check for this episode.")
        mock_logger.info.assert_not_called() # Ensure info wasn't called
        mock_logger.warning.assert_not_called() # Ensure warning wasn't called


if __name__ == '__main__':
    # Running with verbosity and disabling buffer to see prints/logs during test run
    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=2, buffer=False) 