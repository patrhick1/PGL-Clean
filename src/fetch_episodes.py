import logging
from airtable_service import PodcastService
from external_api_service import PodscanFM
from data_processor import parse_date
from datetime import datetime, timedelta
import threading
from typing import Optional, List, Dict, Any
from collections import defaultdict
import requests
from bs4 import BeautifulSoup
from email.utils import parsedate_to_datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import os  # Added for environment variable configuration


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
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

class RateLimitedAirtable:
    """Thin wrapper that ensures no more than *AIRTABLE_MAX_CONCURRENCY* Airtable
    requests happen at the same time.  All public methods of
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


def parse_rss_date(date_string):
    """
    Parse date strings commonly found in RSS feeds.
    Uses email.utils.parsedate_to_datetime as a primary method.
    Falls back to the more general parse_date if needed.
    """
    if not date_string:
        return None
    try:
        # Primarily use email.utils parser, better suited for RSS pubDate formats
        dt = parsedate_to_datetime(date_string)
        # Make naive if timezone conversion fails or is not needed simpler
        if dt.tzinfo:
             # Convert to UTC then make naive for consistent comparison if needed,
             # or handle timezone properly based on requirements.
             # For simplicity here, we might just make it naive after potential conversion
             # Example: dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
             pass # Keep timezone info for now if parsedate_to_datetime handles it
        return dt
    except Exception:
        # Fallback to the general parser if email.utils fails
        logger.warning(f"parsedate_to_datetime failed for '{date_string}', falling back.")
        return parse_date(date_string) # Use the existing robust parser

def fetch_episodes_from_rss(rss_url: str, max_episodes: int = 10, stop_flag: Optional[threading.Event] = None) -> List[Dict[str, Any]]:
    """
    Extracts episode information from the provided RSS feed URL.
    Limits to a maximum number of recent episodes.

    Args:
        rss_url: The URL of the RSS feed.
        max_episodes: The maximum number of episodes to fetch.
        stop_flag: Optional threading event to signal stoppage.

    Returns:
        A list of dictionaries, each containing details of an episode.
        Keys match the structure expected by the main processing logic
        (e.g., "episode_title", "episode_description", "episode_id", "episode_audio_url", "posted_at").
    """
    logger.info(f"Attempting to fetch episodes from RSS feed: {rss_url}")
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
    Creates records in Airtable immediately after processing each podcast.
    Limited to the 10 most recent episodes per podcast within the last 90 days.

    Args:
        stop_flag: Optional threading.Event that signals when to stop processing
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

    table = 'Podcasts'
    batch_size = 10
    max_episodes_per_podcast = 10

    total_episodes_created = 0
    total_podcasts_processed = 0

    # Unpack the Airtable record
    record_id = podcast_record['id']
    podcast_field = podcast_record['fields']
    podcast_name = podcast_field.get('Podcast Name', '[No Name]')
    podcast_id = podcast_field.get('Podcast id')  # Podscan ID
    rss_url = podcast_field.get('RSS Feed')

    logger.info(f"--- Processing podcast: '{podcast_name}' (Airtable ID: {record_id}) ---")

    podcast_episodes: List[Dict[str, Any]] = []
    fetched_from_podscan = False
    source_used = "None"  # Track which source was used

    try:
        # Stop flag check (early)
        if stop_flag and stop_flag.is_set():
            logger.info(f"Stop flag set before starting podcast '{podcast_name}'.")
            return 0, 0

        # -------------------- 1. Fetch from Podscan (if ID present) --------------------
        if podcast_id:
            logger.info(f"Attempting to fetch episodes from Podscan (ID: {podcast_id})")
            try:
                if stop_flag and stop_flag.is_set():
                    return 0, 0

                podcast_episodes = podscan.get_podcast_episode(podcast_id)
                if podcast_episodes:
                    fetched_from_podscan = True
                    source_used = "Podscan"
                    logger.info(f"Successfully fetched {len(podcast_episodes)} episodes from Podscan for '{podcast_name}'")
                else:
                    logger.info(f"Podscan fetch returned no episodes for '{podcast_name}'.")
            except Exception as e:
                logger.warning(f"Error fetching from Podscan for {podcast_name} (ID: {podcast_id}): {e}. Trying RSS fallback.")
                podcast_episodes = []  # Allow RSS fallback

        # -------------------- 2. Fallback to RSS --------------------
        if not fetched_from_podscan:
            if rss_url:
                if stop_flag and stop_flag.is_set():
                    return 0, 0

                logger.info(f"Attempting to fetch episodes from RSS feed for '{podcast_name}'")
                podcast_episodes = fetch_episodes_from_rss(rss_url, max_episodes=max_episodes_per_podcast, stop_flag=stop_flag)
                if podcast_episodes:
                    source_used = "RSS"
                    logger.info(f"Successfully fetched {len(podcast_episodes)} episodes from RSS for '{podcast_name}'")
                else:
                    logger.info(f"RSS fetch returned no episodes for '{podcast_name}'.")
            else:
                logger.warning(f"No Podscan ID and no RSS Feed URL found for '{podcast_name}' (Record ID: {record_id}). Cannot fetch episodes.")

        # -------------------- 3. Handle empty results --------------------
        if not podcast_episodes:
            try:
                airtable.update_record(table, record_id, {"Fetched": True, "Checked": True})
                logger.info(f"✅ Marked podcast '{podcast_name}' as fetched (no episodes found).")
                total_podcasts_processed += 1
            except Exception as e:
                logger.error(f"Error marking podcast '{podcast_name}' as fetched (no episodes): {e}")
            return 0, total_podcasts_processed

        # Skip podcasts that have fewer than 10 episodes total
        if len(podcast_episodes) < 10:
            logger.info(f"Skipping podcast '{podcast_name}' (Source: {source_used}) because it has only {len(podcast_episodes)} episodes (<10).")
            airtable.update_record(table, record_id, {"Checked": True})
            return 0, 0

        # -------------------- 4. Sort + limit episodes --------------------
        sorted_episodes = sorted(
            podcast_episodes,
            key=lambda x: parse_date(x.get("posted_at")) if isinstance(x.get("posted_at"), str) else x.get("posted_at") or datetime.min,
            reverse=True,
        )
        episodes_to_process = sorted_episodes[:max_episodes_per_podcast]

        episodes_to_create: List[Dict[str, Any]] = []

        # -------------------- 5. Episode loop --------------------
        for index, episode in enumerate(episodes_to_process):
            if stop_flag and stop_flag.is_set():
                return 0, 0

            episode_title = episode.get('episode_title', '[No Title]')
            episode_audio_url = episode.get("episode_audio_url")

            # Duplicate check using Episode URL
            if episode_audio_url:
                escaped_episode_url = episode_audio_url.replace("'", "\\'")
                formula = f"AND({{Episode URL}}='{escaped_episode_url}', RECORD_ID({{Podcast}}) = '{record_id}')"
                try:
                    existing = airtable.search_records('Podcast_Episodes', formula=formula)
                    if existing:
                        logger.info(f"Duplicate found for Episode URL '{episode_audio_url}' (Podcast: '{podcast_name}'). Skipping.")
                        continue
                except Exception as e:
                    logger.error(f"Error checking duplicate for '{episode_audio_url}' in '{podcast_name}': {e}")
            else:
                logger.warning(f"Episode '{episode_title}' for podcast '{podcast_name}' lacks audio URL – skipping duplicate check.")

            # Date filtering (keep <=200 days old)
            dt_published_input = episode.get("posted_at")
            dt_published = parse_date(dt_published_input) if isinstance(dt_published_input, str) else dt_published_input
            if not dt_published:
                logger.warning(f"Could not parse date for episode '{episode_title}'. Skipping.")
                continue

            if dt_published < datetime.now(dt_published.tzinfo) - timedelta(days=200):
                logger.info(f"Episode '{episode_title}' ({dt_published.date()}) older than 200 days – skipping.")
                continue

            # Build Airtable payload
            field_to_update = {
                "Episode Title": episode_title,
                "Summary": episode.get('episode_description'),
                "Episode ID": episode.get("episode_id"),
                "Episode URL": episode_audio_url,
                "Episode Web Link": episode.get("episode_url"),
                "Published": dt_published.strftime('%Y-%m-%d'),
                "Podcast": [record_id],
                "Downloaded": fetched_from_podscan and bool(episode.get("episode_transcript")),
            }

            # Transcript handling & flagging
            if fetched_from_podscan and episode.get("episode_transcript"):
                field_to_update["Transcription"] = episode["episode_transcript"]
                field_to_update["Transcribe"] = False
            else:
                field_to_update["Transcribe"] = index < 4  # Flag top 4 for transcription

            episodes_to_create.append(field_to_update)

        # -------------------- 6. Batch create episodes --------------------
        if episodes_to_create:
            logger.info(f"Creating {len(episodes_to_create)} episodes for '{podcast_name}' in batches of {batch_size}")

            for i in range(0, len(episodes_to_create), batch_size):
                if stop_flag and stop_flag.is_set():
                    return total_episodes_created, total_podcasts_processed

                batch = episodes_to_create[i : i + batch_size]
                try:
                    created_records = airtable.create_records_batch('Podcast_Episodes', batch)
                    total_episodes_created += len(created_records)
                except Exception as e:
                    logger.error(f"Error creating batch of episodes for '{podcast_name}': {e}")

        # -------------------- 7. Mark podcast as fetched --------------------
        try:
            airtable.update_record(table, record_id, {"Fetched": True, "Checked": True})
            total_podcasts_processed += 1
            logger.info(f"✅ Marked podcast '{podcast_name}' as fetched (Source: {source_used})")
        except Exception as e:
            logger.error(f"Error marking podcast '{podcast_name}' as fetched: {e}")

    except Exception as e:
        logger.error(f"Unhandled error processing podcast '{podcast_name}' (Record ID: {record_id}): {e}", exc_info=True)

    return total_episodes_created, total_podcasts_processed

if __name__ == "__main__":
     print("Running fetch_episodes...")
     get_podcast_episodes()
     print("fetch_episodes finished.")
