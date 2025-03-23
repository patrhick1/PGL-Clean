# app/external_api_service.py

import os
from dotenv import load_dotenv
import requests
import html

load_dotenv()

class PodscanFM:
    def __init__(self):
        self.api_key = os.getenv("PODSCANAPI")
        self.base_url = 'https://podscan.fm/api/v1'

    def get_category(self):
            # Build the API endpoint URL using the keyword
        url = f'{self.base_url}/categories'

        # Set headers if an API key is provided
        headers = {}
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            # Make the GET request to the API endpoint
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Check for HTTP errors
        except requests.RequestException as error:
            print(f"Error while fetching podcasts: {error}")
            return []

        # Parse JSON response
        data = response.json()

        # Adjust based on the actual response format –
        # if podcasts are nested within a key (e.g., "podcasts"), extract them
        categories = data.get("categories", data)

        # Extract the key parameters for each podcast listing
        results = []
        for category in categories:
            details = {
                "category_id": category.get("category_id"),
                "category_name": category.get("category_name"),
                "category_display_name": category.get("category_display_name")
            }
            results.append(details)

        return results

    def get_podcast_episode(self, podcast_id):
            # Build the API endpoint URL using the keyword
        url = f'{self.base_url}/podcasts/{podcast_id}/episodes'

        # Set headers if an API key is provided
        headers = {"Authorization": f"Bearer {self.api_key}"}
        params = {
            'order_by': 'posted_at',
            'order_dir': 'desc',
            'per_page': 10
        }

        try:
            # Make the GET request to the API endpoint
            response = requests.get(url, headers=headers, params= params)
            response.raise_for_status()  # Check for HTTP errors
        except requests.RequestException as error:
            print(f"Error while fetching podcasts: {error}")
            return []

        # Parse JSON response
        data = response.json()

        # Adjust based on the actual response format –
        # if podcasts are nested within a key (e.g., "podcasts"), extract them
        episodes = data.get("episodes", data)

        # Extract the key parameters for each podcast listing
        results = []
        for episode in episodes:
            details = {
                "episode_id": episode.get("episode_id"),
                "episode_url": episode.get("episode_url"),
                "episode_title": episode.get("episode_title"),
                "episode_audio_url": episode.get("episode_audio_url"),
                "posted_at": episode.get("posted_at"),
                "episode_transcript": episode.get("episode_transcript"),
                'episode_description': episode.get('episode_description')
            }
            results.append(details)

        return results


    def search_podcasts(self, keyword, category_id=None, page=None ):
        """
        Searches podcasts using provided category id via the Podscan API.

        Parameters:
        category_id (str): The search term to query podcasts.
        api_key (str, optional): API key if authentication is required.
        """
        # Build the API endpoint URL using the keyword
        url = f'{self.base_url}/podcasts/search'


        # Set headers if an API key is provided
        headers = {}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        params = {
            'query': keyword,
            'per_page': 20,
            'language': 'en'
        }

        if page:
            params['page'] = page

        if category_id:
            params['category_ids'] = category_id


        try:
            # Make the GET request to the API endpoint
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()  # Check for HTTP errors
        except requests.RequestException as error:
            print(f"Error while fetching podcasts: {error}")
            return []

        # Parse JSON response
        data = response.json()

        podcasts = data.get("podcasts", data)

        # Extract the key parameters for each podcast listing
        results = []
        for podcast in podcasts:
            details = {
                "podcast_id": podcast.get("podcast_id"),
                "podcast_name": html.unescape(podcast.get("podcast_name")),
                "podcast_url": podcast.get("podcast_url"),
                "podcast_description": podcast.get("podcast_description"),
                "email":podcast.get('reach', {}).get('email'),
                "rss_url": podcast.get("rss_url"),
                "last_posted_at": podcast.get("last_posted_at")
            }
            results.append(details)

        return results



class ListenNote:
    def __init__(self):
        self.api_key = os.getenv('LISTEN_NOTES_API_KEY')
        self.base_url = 'https://listen-api.listennotes.com/api/v2'

    def search_podcasts(self, query, genre_ids, offset=0, published_after=None):
        url = f'{self.base_url}/search'
        headers = {
            'X-ListenAPI-Key': self.api_key
        }
        params = {
            'q': query,
            'sort_by_date': 1,
            'page_size': 10,
            'language': 'English',
            'type': 'podcast',
            'offset': offset,
            'interviews_only': 1,
            'genre_ids': genre_ids
        }
        if published_after:
            params['published_after'] = published_after

        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Listen Notes API Error: {response.status_code} {response.text}")
        

class InstantlyAPI:
    def __init__(self):
        self.base_url = 'https://api.instantly.ai/api/v1'

    def add_lead(self, data):
        url = f'{self.base_url}/lead/add'
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.post(url, json=data, headers=headers)
        return response