import logging
from airtable_service import PodcastService
from external_api_service import PodscanFM
from data_processor import parse_date # Assuming this is a robust date parser
from datetime import datetime, timedelta, timezone # Added timezone for robust date comparisons
import threading
from typing import Optional, List, Dict, Any
from collections import defaultdict
import requests
from bs4 import BeautifulSoup
from email.utils import parsedate_to_datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

airtable_client = PodcastService()
podscan_service = PodscanFM()

# ---------------------------------------------------------------------------
# Concurrency & Rate-limiting configuration
# ---------------------------------------------------------------------------
# Maximum number of podcasts to process concurrently. Can be overridden with
# an env-var so we can tune easily in production without code changes.
MAX_FETCH_WORKERS: int = int(os.getenv("FETCH_EPISODE_WORKERS", "40"))

# Airtable has a hard limit of ~5 requests/second per base. We protect all
# Airtable calls with a semaphore so, even though we may spin up many worker
# threads for network-bound RSS fetches, we will never exceed this parallel
# quota when talking to Airtable.
AIRTABLE_MAX_CONCURRENCY: int = int(os.getenv("AIRTABLE_MAX_CONCURRENCY", "5"))

# Binary semaphore used by a lightweight wrapper around PodcastService.
airtable_semaphore = threading.Semaphore(AIRTABLE_MAX_CONCURRENCY)

# --- HTTP request headers ----------------------------------------------------
# Custom headers to mimic a browser when fetching RSS feeds. Some hosts block
# requests that look like bots; using standard browser headers avoids that.
custom_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,application/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

class RateLimitedAirtable:
    """Thin wrapper that ensures no more than *AIRTABLE_MAX_CONCURRENCY* Airtable
    requests happen at the same time. All public methods of
    :class:`~airtable_service.PodcastService` are proxied through a semaphore.
    """

    def __init__(self, airtable_client: PodcastService, semaphore: threading.Semaphore):
        self._client = airtable_client
        self._sem = semaphore

    def __getattr__(self, item):
        # Proxy all attribute access to the underlying client, wrapping callables
        # with the semaphore so we rate-limit only actual HTTP calls.
        attr = getattr(self._client, item)
        if callable(attr):
            def _wrapped(*args, **kwargs):
                with self._sem:
                    return attr(*args, **kwargs)
            return _wrapped
        return attr


def parse_rss_date(date_string: Optional[str]) -> Optional[datetime]:
    """
    Parse date strings commonly found in RSS feeds.
    Uses email.utils.parsedate_to_datetime as a primary method.
    Falls back to the more general parse_date if needed.
    Ensures the returned datetime is timezone-aware and converted to UTC,
    then made naive for consistent comparison with other naive datetimes.
    """
    if not date_string:
        return None
    try:
        dt = parsedate_to_datetime(date_string)
        # Convert to UTC and then make naive for consistent comparison
        if dt.tzinfo:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        logger.warning(f"parsedate_to_datetime failed for '{date_string}', falling back to parse_date.")
        # Assume parse_date also returns a datetime object, ideally UTC naive or aware
        dt_fallback = parse_date(date_string)
        if dt_fallback and dt_fallback.tzinfo:
            dt_fallback = dt_fallback.astimezone(timezone.utc).replace(tzinfo=None)
        return dt_fallback

def fetch_episodes_from_rss(rss_url: str, max_episodes: int = 30, stop_flag: Optional[threading.Event] = None) -> List[Dict[str, Any]]:
    """
    Extracts episode information from the provided RSS feed URL.
    Limits to a maximum number of recent episodes.

    Args:
        rss_url: The URL of the RSS feed.
        max_episodes: The maximum number of episodes to fetch. (Increased default to 30)
        stop_flag: Optional threading event to signal stoppage.

    Returns:
        A list of dictionaries, each containing details of an episode.
        Keys match the structure expected by the main processing logic
        (e.g., "episode_title", "episode_description", "episode_id", "episode_audio_url", "posted_at").
    """
    logger.info(f"Attempting to fetch {max_episodes} episodes from RSS feed: {rss_url}")
    episodes = []
    try:
        # Check stop flag before network request
        if stop_flag and stop_flag.is_set():
            logger.info(f"Stopping RSS fetch for {rss_url} due to stop flag")
            return []

        response = requests.get(rss_url, headers=custom_headers, timeout=15) # Added timeout
        response.raise_for_status() # Check for HTTP errors first

        # Use 'lxml-xml' or 'xml' for parsing XML
        soup = BeautifulSoup(response.content, 'xml')
        items = soup.find_all('item')
        logger.info(f"Found {len(items)} items in RSS feed: {rss_url}")

        parsed_episodes = []
        for item in items:
            # Check stop flag inside item processing loop
            if stop_flag and stop_flag.is_set():
                logger.info(f"Stopping RSS item processing for {rss_url} due to stop flag")
                return [] # Return empty list as processing was interrupted

            pub_date_str = item.find('pubDate').text if item.find('pubDate') else None
            pub_date = parse_rss_date(pub_date_str)

            # Try finding audio URL in enclosure, or fallback to guid if it's a URL
            audio_url = None
            if item.find('enclosure') and item.find('enclosure').get('url'):
                audio_url = item.find('enclosure')['url']
            # Basic check if guid might be an audio url (less reliable)
            elif item.find('guid') and item.find('guid').text.startswith(('http://', 'https://')):
                 guid_text = item.find('guid').text
                 # Simple check for common audio extensions
                 if any(guid_text.lower().endswith(ext) for ext in ['.mp3', '.m4a', '.ogg', '.wav']):
                      audio_url = guid_text

            details = {
                # Use keys consistent with Podscan data structure for easier processing
                "episode_title": item.find('title').text if item.find('title') else 'No Title',
                "episode_description": item.find('description').text if item.find('description') else 'No Description',
                "episode_id": item.find('guid').text if item.find('guid') else None, # Keep original GUID
                "episode_audio_url": audio_url, # Use extracted audio URL
                "posted_at": pub_date, # Store datetime object for sorting
                "episode_url": item.find('link').text if item.find('link') else None, # Web link if available
                "episode_transcript": None, # RSS feeds typically don't have transcripts
            }
            # Only add if we could parse the date, essential for sorting and filtering
            if details["posted_at"]:
                parsed_episodes.append(details)
            else:
                 logger.warning(f"Skipping episode '{details['episode_title']}' due to unparsable date: {pub_date_str}")


        # Sort by parsed date (newest first)
        parsed_episodes.sort(key=lambda x: x["posted_at"], reverse=True)

        # Return the top N episodes
        episodes = parsed_episodes[:max_episodes]
        logger.info(f"Successfully parsed and sorted {len(episodes)} episodes from RSS: {rss_url}")

    except requests.RequestException as e:
        logger.error(f"Failed to fetch or process RSS feed {rss_url}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while processing RSS feed {rss_url}: {e}")

    return episodes

# --- End RSS Fetching Logic ---


def get_podcast_episodes(stop_flag: Optional[threading.Event] = None):
    """
    Fetches podcast episodes, prioritizing PodScan if available, otherwise using RSS feed.
    Ensures only the latest 10 episodes are maintained in Airtable for each podcast,
    deleting older ones if necessary. Manages transcription flags.
    """
    view = "Not Fetched"
    table = 'Podcasts'

    # Fetch all podcasts to process at once (rate-limited)
    podcast_records = RateLimitedAirtable(airtable_client, airtable_semaphore).search_records(table, view=view)
    logger.info(f"Found {len(podcast_records)} podcasts to process from '{view}' view")

    total_episodes_created = 0
    total_podcasts_processed = 0

    # Use a thread-pool to process each podcast concurrently
    with ThreadPoolExecutor(max_workers=min(MAX_FETCH_WORKERS, len(podcast_records))) as executor:
        futures = [executor.submit(_process_single_podcast, rec, stop_flag) for rec in podcast_records]

        for future in as_completed(futures):
            try:
                eps, pods = future.result()
                total_episodes_created += eps
                total_podcasts_processed += pods
            except Exception as exc:
                logger.error(f"Podcast processing task raised an exception: {exc}")

    logger.info(f"🎉 Completed processing! Created {total_episodes_created} episodes across {total_podcasts_processed} podcasts processed.")

def _process_single_podcast(podcast_record: Dict[str, Any], stop_flag: Optional[threading.Event] = None) -> tuple[int, int]:
    """Process one podcast record (fetch episodes + Airtable writes).

    Returns
    -------
    tuple  (episodes_created, podcasts_processed)
    """
    # Thread-local service instances (safer than sharing a single client)
    airtable = RateLimitedAirtable(PodcastService(), airtable_semaphore)
    podscan = PodscanFM()

    podcast_table = 'Podcasts'
    episodes_table = 'Podcast_Episodes'
    batch_size = 10
    max_episodes_to_keep = 10 # The target number of episodes to maintain in Airtable
    rss_fetch_limit = 30 # Fetch more from RSS to ensure we get the latest 10

    total_episodes_created_for_podcast = 0
    podcast_processed_flag = 0 # 1 if processed, 0 if skipped/error

    # Unpack the Airtable record
    record_id = podcast_record['id']
    podcast_field = podcast_record['fields']
    podcast_name = podcast_field.get('Podcast Name', '[No Name]')
    podcast_id = podcast_field.get('Podcast id')  # Podscan ID
    rss_url = podcast_field.get('RSS Feed')

    logger.info(f"--- Processing podcast: '{podcast_name}' (Airtable ID: {record_id}) ---")

    all_fetched_episodes: List[Dict[str, Any]] = [] # Will store all episodes fetched from external sources
    fetched_from_podscan = False
    source_used = "None"

    try:
        # Stop flag check (early)
        if stop_flag and stop_flag.is_set():
            logger.info(f"Stop flag set before starting podcast '{podcast_name}'.")
            return 0, 0

        # -------------------- 1. Fetch from Podscan (if ID present) --------------------
        if podcast_id:
            logger.info(f"Attempting to fetch episodes from Podscan (ID: {podcast_id})")
            try:
                if stop_flag and stop_flag.is_set(): return 0, 0

                podscan_episodes = podscan.get_podcast_episode(podcast_id)
                if podscan_episodes:
                    all_fetched_episodes.extend(podscan_episodes)
                    fetched_from_podscan = True
                    source_used = "Podscan"
                    logger.info(f"Successfully fetched {len(podscan_episodes)} episodes from Podscan for '{podcast_name}'")
                else:
                    logger.info(f"Podscan fetch returned no episodes for '{podcast_name}'.")
            except Exception as e:
                logger.warning(f"Error fetching from Podscan for {podcast_name} (ID: {podcast_id}): {e}. Trying RSS fallback.")
                # Do NOT clear all_fetched_episodes here if Podscan failed,
                # as we might still have some partial data or want to combine with RSS.
                # For this specific case, if Podscan fails, we treat it as if it returned nothing
                # and rely on RSS. So, clearing is appropriate if Podscan is the primary source.
                all_fetched_episodes = [] # Clear if Podscan failed, so RSS can populate

        # -------------------- 2. Fallback to RSS --------------------
        # Always try RSS if Podscan failed or didn't provide enough, or if no Podscan ID
        # We fetch more from RSS to ensure we have enough candidates for the latest 10
        if not all_fetched_episodes or len(all_fetched_episodes) < max_episodes_to_keep:
            if rss_url:
                if stop_flag and stop_flag.is_set(): return 0, 0

                logger.info(f"Attempting to fetch episodes from RSS feed for '{podcast_name}'")
                rss_episodes = fetch_episodes_from_rss(rss_url, max_episodes=rss_fetch_limit, stop_flag=stop_flag)
                if rss_episodes:
                    all_fetched_episodes.extend(rss_episodes)
                    if not fetched_from_podscan: # Only set source if Podscan wasn't primary
                        source_used = "RSS"
                    logger.info(f"Successfully fetched {len(rss_episodes)} episodes from RSS for '{podcast_name}'")
                else:
                    logger.info(f"RSS fetch returned no episodes for '{podcast_name}'.")
            else:
                logger.warning(f"No Podscan ID and no RSS Feed URL found for '{podcast_name}' (Record ID: {record_id}). Cannot fetch episodes.")

        # -------------------- 3. Handle cases where no episodes are found from any source --------------------
        if not all_fetched_episodes:
            try:
                airtable.update_record(podcast_table, record_id, {"Fetched": True, "Checked": True})
                logger.info(f"✅ Marked podcast '{podcast_name}' as fetched (no episodes found).")
                podcast_processed_flag = 1
            except Exception as e:
                logger.error(f"Error marking podcast '{podcast_name}' as fetched (no episodes): {e}")
            return 0, podcast_processed_flag

        # -------------------- 4. Fetch existing episodes from Airtable --------------------
        existing_airtable_episodes = []
        existing_episode_map = {} # Map episode_url -> airtable_record (for quick lookup of original Airtable data)
        try:
            # Fetch all episodes linked to this podcast
            formula = f"{{Podcast}} = '{record_id}'"
            existing_airtable_episodes = airtable.search_records(episodes_table, formula=formula)
            logger.info(f"Found {len(existing_airtable_episodes)} existing episodes in Airtable for '{podcast_name}'.")
            for rec in existing_airtable_episodes:
                episode_url = rec['fields'].get('Episode URL')
                if episode_url:
                    existing_episode_map[episode_url] = rec
        except Exception as e:
            logger.error(f"Error fetching existing episodes for '{podcast_name}' from Airtable: {e}")

        # -------------------- 5. Consolidate, Deduplicate, and Sort all episodes (PRIORITIZING AIRTABLE) --------------------
        master_episode_list = []
        episode_url_to_master_index = {} # Map episode_url to its index in master_episode_list

        # 5.1. Populate master_episode_list with existing Airtable data (PRIORITY)
        for rec in existing_airtable_episodes:
            episode_url = rec['fields'].get('Episode URL')
            if not episode_url:
                logger.warning(f"Existing Airtable episode '{rec['id']}' has no Episode URL. Skipping for master list.")
                continue

            posted_at_str = rec['fields'].get('Published')
            posted_at_dt = None
            if posted_at_str:
                try:
                    # Airtable date format is YYYY-MM-DD
                    posted_at_dt = datetime.strptime(posted_at_str, '%Y-%m-%d').replace(tzinfo=None)
                except ValueError:
                    logger.warning(f"Could not parse Airtable date '{posted_at_str}' for existing episode '{rec['fields'].get('Episode Title')}'.")

            # Create a base entry from Airtable data
            airtable_episode_data = {
                "episode_title": rec['fields'].get('Episode Title'),
                "episode_description": rec['fields'].get('Summary'),
                "episode_id": rec['fields'].get('Episode ID'),
                "episode_audio_url": episode_url,
                "posted_at": posted_at_dt, # Store datetime object
                "episode_url": rec['fields'].get('Episode Web Link'),
                "episode_transcript": rec['fields'].get('Transcription'), # Keep existing transcript
                "airtable_record_id": rec['id'], # Store Airtable ID for updates/deletes
                "was_fetched_externally": False, # Assume not fetched externally yet
                "was_fetched_from_podscan": False, # Assume not from Podscan yet
            }
            master_episode_list.append(airtable_episode_data)
            episode_url_to_master_index[episode_url] = len(master_episode_list) - 1

        # 5.2. Integrate newly fetched episodes, merging where URLs match
        for new_episode in all_fetched_episodes:
            episode_url = new_episode.get("episode_audio_url")
            if not episode_url:
                logger.warning(f"Skipping fetched episode '{new_episode.get('episode_title', 'N/A')}' due to missing audio URL.")
                continue

            # Standardize date for the new episode
            if isinstance(new_episode.get("posted_at"), str):
                new_episode["posted_at"] = parse_rss_date(new_episode["posted_at"])
            elif isinstance(new_episode.get("posted_at"), datetime) and new_episode["posted_at"].tzinfo:
                new_episode["posted_at"] = new_episode["posted_at"].astimezone(timezone.utc).replace(tzinfo=None)

            if episode_url in episode_url_to_master_index:
                # Episode exists in Airtable (already in master_episode_list), MERGE new data
                master_index = episode_url_to_master_index[episode_url]
                existing_master_entry = master_episode_list[master_index]

                # Prioritize existing Airtable data for most fields, but update if new data is better/missing
                # For transcript: if new_episode has it and existing doesn't, take new.
                if new_episode.get("episode_transcript") and not existing_master_entry.get("episode_transcript"):
                    existing_master_entry["episode_transcript"] = new_episode["episode_transcript"]
                    logger.debug(f"Merged new transcript for '{new_episode.get('episode_title')}' from fetched data.")
                
                # Update other fields if Airtable's is empty or new data is more complete
                if not existing_master_entry.get("episode_title") and new_episode.get("episode_title"):
                    existing_master_entry["episode_title"] = new_episode["episode_title"]
                if not existing_master_entry.get("episode_description") and new_episode.get("episode_description"):
                    existing_master_entry["episode_description"] = new_episode["episode_description"]
                if not existing_master_entry.get("episode_url") and new_episode.get("episode_url"):
                    existing_master_entry["episode_url"] = new_episode["episode_url"]
                if not existing_master_entry.get("episode_id") and new_episode.get("episode_id"):
                    existing_master_entry["episode_id"] = new_episode["episode_id"]
                
                # Always update posted_at if the new one is more precise or different
                if new_episode.get("posted_at") and (not existing_master_entry.get("posted_at") or new_episode["posted_at"] != existing_master_entry["posted_at"]):
                    existing_master_entry["posted_at"] = new_episode["posted_at"]

                # Mark that this episode was seen in the new fetch
                existing_master_entry["was_fetched_externally"] = True
                existing_master_entry["was_fetched_from_podscan"] = fetched_from_podscan # Propagate this flag

            else:
                # This is a truly NEW episode, not yet in Airtable
                new_episode["airtable_record_id"] = None # No Airtable ID yet
                new_episode["was_fetched_externally"] = True
                new_episode["was_fetched_from_podscan"] = fetched_from_podscan
                master_episode_list.append(new_episode)
                episode_url_to_master_index[episode_url] = len(master_episode_list) - 1

        # Sort the master list by date (newest first)
        master_episode_list.sort(key=lambda x: x.get("posted_at") or datetime.min, reverse=True)

        # Filter out episodes older than 200 days *before* selecting the top 10
        final_episodes_to_manage_pre_limit = []
        now_utc_naive = datetime.now(timezone.utc).replace(tzinfo=None)
        for ep in master_episode_list:
            if ep.get("posted_at"):
                # Ensure comparison is between naive datetimes
                if ep["posted_at"] >= now_utc_naive - timedelta(days=200):
                    final_episodes_to_manage_pre_limit.append(ep)
                else:
                    logger.debug(f"Episode '{ep.get('episode_title', 'N/A')}' ({ep['posted_at'].date()}) older than 200 days – excluding from consideration.")
            else:
                logger.warning(f"Episode '{ep.get('episode_title', 'N/A')}' has no valid posted_at date, excluding from consideration.")

        # Now, take the top `max_episodes_to_keep` from the filtered list
        final_episodes_to_manage = final_episodes_to_manage_pre_limit[:max_episodes_to_keep]
        logger.info(f"Identified {len(final_episodes_to_manage)} episodes to manage for '{podcast_name}' (target: {max_episodes_to_keep}).")

        # -------------------- 6. Determine actions: Delete, Create, Update --------------------
        episodes_to_delete_ids = []
        episodes_to_create_payloads = []
        episodes_to_update_payloads = []

        # Identify episodes to delete (existing in Airtable but not in final_episodes_to_manage)
        final_episode_urls_to_keep = {ep.get("episode_audio_url") for ep in final_episodes_to_manage if ep.get("episode_audio_url")}
        for rec_url, rec_data in existing_episode_map.items():
            if rec_url not in final_episode_urls_to_keep:
                episodes_to_delete_ids.append(rec_data['id'])
        
        if episodes_to_delete_ids:
            logger.info(f"Marked {len(episodes_to_delete_ids)} episodes for deletion for '{podcast_name}'.")

        # Identify episodes to create or update
        transcribe_flag_count = 0 # Counter for flagging 'Transcribe'
        for index, episode in enumerate(final_episodes_to_manage):
            if stop_flag and stop_flag.is_set(): return 0, 0

            episode_url = episode.get("episode_audio_url")
            if not episode_url:
                logger.warning(f"Skipping episode '{episode.get('episode_title', 'N/A')}' due to missing audio URL during final processing.")
                continue

            dt_published = episode.get("posted_at")
            # dt_published should already be a naive datetime object from step 5
            if not dt_published:
                logger.warning(f"Could not parse date for episode '{episode.get('episode_title', 'N/A')}' during final processing. Skipping.")
                continue

            # Determine 'Downloaded' and 'Transcribe' status based on the merged data
            has_transcript_in_master = bool(episode.get("episode_transcript"))
            
            # Downloaded is True if it came from Podscan AND has a transcript (either new or existing)
            downloaded = episode.get("was_fetched_from_podscan", False) and has_transcript_in_master
            
            transcribe = False # Start with False for Transcribe, then set based on conditions
            
            if has_transcript_in_master:
                # If there's a transcript, both Downloaded and Transcribe are True
                transcribe = True 
            else: # No transcript
                # If no transcript, flag for transcription only if within the first 4 untranscribed
                if transcribe_flag_count < 4:
                    transcribe = True
                    transcribe_flag_count += 1

            field_to_update = {
                "Episode Title": episode.get('episode_title', 'No Title'),
                "Summary": episode.get('episode_description'),
                "Episode ID": episode.get("episode_id"),
                "Episode URL": episode_url,
                "Episode Web Link": episode.get("episode_url"),
                "Published": dt_published.strftime('%Y-%m-%d') if dt_published else None,
                "Podcast": [record_id],
                "Downloaded": downloaded,
                "Transcribe": transcribe,
            }

            # Handle Transcription field:
            # If the master entry has a transcript, include it.
            if has_transcript_in_master:
                field_to_update["Transcription"] = episode["episode_transcript"]
            else:
                # If no transcript in master entry, explicitly clear it in Airtable
                # This handles cases where a transcript was manually added but then removed from source,
                # or if we switch from Podscan to RSS and the transcript is no longer available.
                field_to_update["Transcription"] = None 

            # Check if episode already exists in Airtable (using the airtable_record_id stored in master_episode_list)
            existing_airtable_record_id = episode.get("airtable_record_id")
            if existing_airtable_record_id:
                # This episode already exists in Airtable, prepare for update
                current_fields = existing_episode_map.get(episode_url, {}).get('fields', {}) # Get original fields for comparison
                update_data = {}
                needs_update = False

                # Compare fields to determine if an update is necessary
                for key, value in field_to_update.items():
                    current_value = current_fields.get(key)
                    if key == "Published":
                        # Compare date strings directly for Airtable format
                        if current_value != value:
                            needs_update = True
                            update_data[key] = value
                    elif key == "Podcast":
                        # Compare list of record IDs (Airtable stores linked records as a list of IDs)
                        if current_value != value: # This will compare lists directly
                            needs_update = True
                            update_data[key] = value
                    elif key == "Transcription":
                        # Special handling for Transcription:
                        # If new value is None and current is not None, it's a change (clear)
                        # If new value is not None and current is different, it's a change (update)
                        if value is None and current_value is not None:
                            needs_update = True
                            update_data[key] = value
                        elif value is not None and current_value != value:
                            needs_update = True
                            update_data[key] = value
                    elif current_value != value:
                        needs_update = True
                        update_data[key] = value
                
                if needs_update:
                    episodes_to_update_payloads.append({"id": existing_airtable_record_id, "fields": update_data})
            else:
                # This is a new episode, prepare for creation
                episodes_to_create_payloads.append(field_to_update)

        # -------------------- 7. Execute Airtable operations --------------------
        # 7.1. Delete excess episodes
        if episodes_to_delete_ids:
            logger.info(f"Deleting {len(episodes_to_delete_ids)} excess episodes for '{podcast_name}'.")
            try:
                airtable.delete_records_batch(episodes_table, episodes_to_delete_ids)
                logger.info(f"Successfully deleted {len(episodes_to_delete_ids)} episodes for '{podcast_name}'.")
            except Exception as e:
                logger.error(f"Error deleting episodes for '{podcast_name}': {e}")

        # 7.2. Create new episodes
        if episodes_to_create_payloads:
            logger.info(f"Creating {len(episodes_to_create_payloads)} new episodes for '{podcast_name}'.")
            for i in range(0, len(episodes_to_create_payloads), batch_size):
                if stop_flag and stop_flag.is_set(): return total_episodes_created_for_podcast, podcast_processed_flag
                batch = episodes_to_create_payloads[i : i + batch_size]
                try:
                    created_records = airtable.create_records_batch(episodes_table, batch)
                    total_episodes_created_for_podcast += len(created_records)
                    logger.info(f"Created batch of {len(created_records)} episodes for '{podcast_name}'.")
                except Exception as e:
                    logger.error(f"Error creating batch of episodes for '{podcast_name}': {e}")

        # 7.3. Update existing episodes
        if episodes_to_update_payloads:
            logger.info(f"Updating {len(episodes_to_update_payloads)} existing episodes for '{podcast_name}'.")
            for i in range(0, len(episodes_to_update_payloads), batch_size):
                if stop_flag and stop_flag.is_set(): return total_episodes_created_for_podcast, podcast_processed_flag
                batch = episodes_to_update_payloads[i : i + batch_size]
                try:
                    updated_records = airtable.update_records_batch(episodes_table, batch)
                    logger.info(f"Updated batch of {len(updated_records)} episodes for '{podcast_name}'.")
                except Exception as e:
                    logger.error(f"Error updating batch of episodes for '{podcast_name}': {e}")

        # -------------------- 8. Mark podcast as fetched --------------------
        try:
            airtable.update_record(podcast_table, record_id, {"Fetched": True, "Checked": True})
            podcast_processed_flag = 1
            logger.info(f"✅ Marked podcast '{podcast_name}' as fetched (Source: {source_used})")
        except Exception as e:
            logger.error(f"Error marking podcast '{podcast_name}' as fetched: {e}")

    except Exception as e:
        logger.error(f"Unhandled error processing podcast '{podcast_name}' (Record ID: {record_id}): {e}", exc_info=True)

    return total_episodes_created_for_podcast, podcast_processed_flag

if __name__ == "__main__":
     print("Running fetch_episodes...")
     get_podcast_episodes()
     print("fetch_episodes finished.")