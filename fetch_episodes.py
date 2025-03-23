import logging
from airtable_service import PodcastService
from external_api_service import PodscanFM
from data_processor import DataProcessor, parse_date
from datetime import datetime, timedelta


# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

airtable_client = PodcastService()
podscan_service = PodscanFM()


def get_podcast_episodes():
    view = "Not Fetched"
    table = 'Podcasts'
    podcast_records = airtable_client.search_records(table, view=view)

    for podcast in podcast_records:
        record_id = podcast['id']
        podcast_field = podcast['fields']
        podcast_id = podcast_field.get('Podcast id')
        podcast_name = podcast_field.get('Podcast Name')

        podcast_episodes = podscan_service.get_podcast_episode(podcast_id)
        logger.info(
            f"Fetched {len(podcast_episodes)} episodes for podcast '{podcast_name}'."
        )

        for episode in podcast_episodes:
            # Parse the published date
            dt_published = parse_date(episode.get("posted_at"))
            if dt_published is None:
                logger.warning(
                    f"Could not parse published date for episode '{episode.get('episode_title')}'. Skipping this episode."
                )
                continue

            # Check if the episode was released within the past year
            one_year_ago = datetime.now(
                dt_published.tzinfo) - timedelta(days=90)
            if dt_published < one_year_ago:
                logger.info(
                    f"Episode '{episode.get('episode_title')}' is older than one year. Skipping update."
                )
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

            airtable_client.create_record('Podcast_Episodes', field_to_update)
            logger.info(
                f"Podcast episode '{episode.get('episode_title')}' created for {podcast_name}."
            )

        airtable_client.update_record(table, record_id, {"Fetched": True})
        logger.info(
            f"Successfully fetched and updated episodes for {podcast_name}.")
