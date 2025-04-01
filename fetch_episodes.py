import logging
from airtable_service import PodcastService
from external_api_service import PodscanFM
from data_processor import DataProcessor, parse_date
from datetime import datetime, timedelta
import threading
from typing import Optional, List, Dict, Any
from collections import defaultdict


# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

airtable_client = PodcastService()
podscan_service = PodscanFM()


def get_podcast_episodes(stop_flag: Optional[threading.Event] = None):
    """
    Fetch podcast episodes from PodScan and create records in Airtable immediately after processing each podcast.
    
    Args:
        stop_flag: Optional threading.Event that signals when to stop processing
    """
    view = "Not Fetched"
    table = 'Podcasts'
    
    # Fetch all podcasts to process at once
    podcast_records = airtable_client.search_records(table, view=view)
    logger.info(f"Found {len(podcast_records)} podcasts to process")
    
    # Track total episodes created for reporting
    total_episodes_created = 0
    total_podcasts_processed = 0
    
    # Batch size for episode creation (per podcast)
    batch_size = 10
    
    # Process each podcast individually
    for podcast in podcast_records:
        # Check if we should stop
        if stop_flag and stop_flag.is_set():
            logger.info("Stopping podcast episode fetching due to stop flag")
            return
            
        record_id = podcast['id']
        podcast_field = podcast['fields']
        podcast_id = podcast_field.get('Podcast id')
        podcast_name = podcast_field.get('Podcast Name')
        
        try:
            # Fetch all episodes for this podcast
            podcast_episodes = podscan_service.get_podcast_episode(podcast_id)
            logger.info(f"Fetched {len(podcast_episodes)} episodes for podcast '{podcast_name}'")
            
            # Collect episodes for this podcast
            episodes_to_create = []
            episodes_for_this_podcast = 0
            
            for episode in podcast_episodes:
                # Check if we should stop
                if stop_flag and stop_flag.is_set():
                    logger.info(f"Stopping episode processing for podcast '{podcast_name}' due to stop flag")
                    return
                    
                # Parse the published date
                dt_published = parse_date(episode.get("posted_at"))
                if dt_published is None:
                    logger.warning(f"Could not parse published date for episode '{episode.get('episode_title')}'. Skipping this episode.")
                    continue
    
                # Check if the episode was released within the past 90 days
                days_ago = datetime.now(dt_published.tzinfo) - timedelta(days=90)
                if dt_published < days_ago:
                    logger.info(f"Episode '{episode.get('episode_title')}' is older than 90 days. Skipping.")
                    continue
    
                # Format the published date as a string
                published_date = dt_published.strftime('%Y-%m-%d')
                transcript = episode.get("episode_transcript")
                field_to_update = {
                    "Episode Title": episode.get("episode_title"),
                    "Summary": episode.get('episode_description'),
                    "Episode ID": episode.get("episode_id"),
                    "Episode URL": episode.get("episode_audio_url"),
                    "Episode Web Link": episode.get("episode_url"),
                    "Published": published_date,
                    "Transcribe": True,
                    "Podcast": [record_id]
                }
                if transcript:
                    field_to_update['Transcription'] = transcript
                    field_to_update['Downloaded'] = True
                
                # Add to this podcast's batch
                episodes_to_create.append(field_to_update)
                episodes_for_this_podcast += 1
            
            # Now create episodes for this podcast in batches
            if episodes_to_create:
                logger.info(f"Creating {len(episodes_to_create)} episodes for podcast '{podcast_name}' in batches of {batch_size}")
                podcast_episodes_created = 0
                
                for i in range(0, len(episodes_to_create), batch_size):
                    # Check if we should stop
                    if stop_flag and stop_flag.is_set():
                        logger.info(f"Stopping batch creation for podcast '{podcast_name}' at index {i} due to stop flag")
                        return
                        
                    # Get the next batch for this podcast
                    batch = episodes_to_create[i:i+batch_size]
                    
                    try:
                        # Create the batch of records
                        created_records = airtable_client.create_records_batch('Podcast_Episodes', batch)
                        podcast_episodes_created += len(created_records)
                        logger.info(f"Created batch of {len(created_records)} episodes for podcast '{podcast_name}'")
                    except Exception as e:
                        logger.error(f"Error creating batch of episodes for podcast '{podcast_name}': {str(e)}")
                
                # Mark this podcast as processed immediately after creating its episodes
                if podcast_episodes_created > 0:
                    try:
                        airtable_client.update_record(table, record_id, {"Fetched": True})
                        logger.info(f"✅ Marked podcast '{podcast_name}' as fetched after creating {podcast_episodes_created} episodes")
                        total_podcasts_processed += 1
                        total_episodes_created += podcast_episodes_created
                    except Exception as e:
                        logger.error(f"Error marking podcast '{podcast_name}' as fetched: {str(e)}")
                else:
                    logger.warning(f"No episodes were successfully created for podcast '{podcast_name}', not marking as fetched")
            else:
                logger.info(f"No valid episodes found for podcast '{podcast_name}', skipping")
                
        except Exception as e:
            logger.error(f"Error processing podcast '{podcast_name}': {str(e)}")
            # Continue to the next podcast
    
    # Log final summary
    logger.info(f"🎉 Completed processing! Created {total_episodes_created} episodes across {total_podcasts_processed} podcasts")
