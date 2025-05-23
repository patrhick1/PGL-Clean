# app/external_api_service.py

import os
from dotenv import load_dotenv
import requests
import html
import pprint # Import pprint

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

    def search_podcast_by_rss(self, rss_feed_url):
        """
        Searches for podcasts by RSS feed URL via the Podscan API.

        Parameters:
        rss_feed_url (str): The RSS feed URL of the podcast.
        """
        # Build the API endpoint URL
        url = f'{self.base_url}/podcasts/search/by/RSS'

        # Set headers
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # Set parameters
        params = {'rss_feed': rss_feed_url}

        try:
            # Make the GET request to the API endpoint
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()  # Check for HTTP errors
        except requests.RequestException as error:
            # Print the error and return an empty list for consistency
            print(f"Error while fetching podcast by RSS: {error}")
            return [] 

        # Parse JSON response safely
        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError as json_err:
            print(f"Error decoding JSON response for RSS {rss_feed_url}: {json_err}")
            print(f"Response content: {response.text}") # Log the raw response
            return [] # Return empty list on JSON decode error

        # Check if the parsed data is actually a list
        if not isinstance(data, list):
            print(f"Unexpected response format for RSS {rss_feed_url}. Expected list, got {type(data)}")
            print(f"Response data: {data}")
            return [] # Return empty list if not a list

        podcasts = data

        # Extract the key parameters for each podcast listing
        results = []
        for podcast in podcasts:
            details = {
                "podcast_id": podcast.get("podcast_id"),
                "podcast_name": html.unescape(podcast.get("podcast_name")),
                "podcast_url": podcast.get("podcast_url"),
                "podcast_description": podcast.get("podcast_description"),
                "email": podcast.get('reach', {}).get('email'),
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
            'episode_count_min': 10,
            'type': 'podcast',
            'offset': offset,
            'region': 'US',
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
        self.base_url = "https://api.instantly.ai/api/v2"

    def add_lead(self, data):
        url = f'{self.base_url}/lead/add'
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.post(url, json=data, headers=headers)
        return response
    
    def add_lead_v2(self, data):
        url = f'{self.base_url}/leads'       
        headers = {
            "Authorization": f"Bearer {os.getenv('INSTANTLY_API_KEY')}",
            "Content-Type": "application/json"
        }
        response = requests.post(url, headers=headers, data=data)
        return response
    
    def list_campaigns(self):
        """
        Lists campaigns from Instantly API.
        Mainly used to test API key validity.
        """
        url = f'{self.base_url}/campaigns'
        headers = {
            "Authorization": f"Bearer {os.getenv('INSTANTLY_API_KEY')}"
        }
        params = {
            "limit": "1"  # Fetch only 1 campaign to minimize data transfer
        }
        try:
            response = requests.get(url, headers=headers, params=params)
            return response
        except requests.RequestException as e:
            # Log or handle the exception as appropriate for your application
            print(f"Error during Instantly API list_campaigns request: {e}")
            # Return a mock response or re-raise if critical
            # For this test, we'll let the caller handle None or check status_code on a mock
            return None # Or a requests.Response object with an error status

    def list_emails(self, limit: int = 100, starting_after: str = None):
        """
        Fetches a list of emails from the Instantly API /emails endpoint.

        Args:
            limit (int): The number of emails to fetch per page.
            starting_after (str, optional): The cursor for pagination to fetch the next set of emails.

        Returns:
            dict: The JSON response from the API (containing 'items' and 'next_starting_after'),
                  or None if an error occurs.
        """
        api_key = os.getenv("INSTANTLY_API_KEY")
        if not api_key:
            print("Error: INSTANTLY_API_KEY not found in environment variables.")
            return None

        endpoint_url = f"{self.base_url}/emails"
        
        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        params = {
            "limit": limit,
        }
        if starting_after:
            params["starting_after"] = starting_after

        try:
            response = requests.get(endpoint_url, headers=headers, params=params, timeout=20)
            response.raise_for_status()  # Stop on API error (4xx or 5xx response)
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred while fetching emails: {http_err} - {response.text}")
        except requests.exceptions.RequestException as req_err:
            print(f"Request error occurred while fetching emails: {req_err}")
        except ValueError as json_err: # Includes JSONDecodeError
            print(f"JSON decode error occurred while fetching emails: {json_err} - Response: {response.text if 'response' in locals() else 'No response object'}")
        return None

    def list_leads_from_campaign(self, campaign_id: str, search: str = None):
        """
        Fetches all leads from a specific Instantly campaign.

        Args:
            campaign_id (str): The UUID of the campaign.

        Returns:
            list: A list of all leads in the campaign, or an empty list if an error occurs or no leads are found.
        """
        api_key = os.getenv("INSTANTLY_API_KEY")
        if not api_key:
            print("Error: INSTANTLY_API_KEY not found in environment variables.")
            return []
        if not campaign_id:
            print("Error: campaign_id must be provided.")
            return []

        endpoint_url = "https://api.instantly.ai/api/v2/leads/list" 
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json"
        }

        limit = 100  # Max allowed by Instantly API, updated from 10
        all_leads = []
        starting_after = None # For pagination

        while True:
            payload = {
                "campaign": campaign_id, # Corrected from "campaign_id"
                "limit": limit,
            }
            if starting_after:
                payload["starting_after"] = starting_after
            if search:
                payload["search"] = search
            try:
                response = requests.post(endpoint_url, json=payload, headers=headers)
                response.raise_for_status()  # Stop on API error (4xx or 5xx response)
                data = response.json()
            except requests.exceptions.HTTPError as http_err:
                print(f"HTTP error occurred while fetching leads: {http_err} - {response.text}")
                break 
            except requests.exceptions.RequestException as req_err:
                print(f"Request error occurred while fetching leads: {req_err}")
                break 
            except ValueError as json_err: # Includes JSONDecodeError
                print(f"JSON decode error occurred while fetching leads: {json_err} - Response: {response.text}")
                break

            leads_this_page = data.get("items", []) # Corrected from "data" to "items"
            all_leads.extend(leads_this_page)

            next_starting_after = data.get("next_starting_after") # Get cursor for next page

            # Done if there's no next page cursor or if the current page wasn't full
            if not next_starting_after or len(leads_this_page) < limit:
                break
            
            starting_after = next_starting_after # Set for the next iteration
        
        return all_leads

if __name__ == "__main__":
    # podscan_fm = PodscanFM()
    # episodes = podscan_fm.get_podcast_episode('pd_dpmk29nqqza9ev8n')
    # for episode in episodes:
    #     print(f'Episode Title: {episode["episode_title"]}')
    #     print(f'Episode Description: {episode["episode_description"]}')
    #     print(f'Episode URL: {episode["episode_url"]}')
    #     print(f'Episode Audio URL: {episode["episode_audio_url"]}')
    #     print(f'Episode Transcript: {episode["episode_transcript"][:500]}')
    #     print(f'Episode Posted At: {episode["posted_at"]}')
    #     print('--------------------------------')

    print("\n--- Testing InstantlyAPI: List Leads from Campaign ---")
    instantly_service = InstantlyAPI()
    test_campaign_id = "b55c61b6-262c-4390-b6e0-63dfca1620c2"
    search = "jill@thedynamiccommunicator.com"
    
    # First, let's test if the API key is valid by listing campaigns (optional, but good for diagnostics)
    print(f"Attempting to list campaigns (API key test)...")
    campaigns_response = instantly_service.list_campaigns()
    if campaigns_response:
        if campaigns_response.status_code == 200:
            print(f"Successfully connected to Instantly API. Status: {campaigns_response.status_code}")
            # campaign_data = campaigns_response.json()
            # print(f"Found campaigns (first page sample): {campaign_data.get('data', [])[:1]}") # Print a sample
        else:
            print(f"Error connecting to Instantly API or invalid API key. Status: {campaigns_response.status_code}, Response: {campaigns_response.text}")
            print("Skipping list_leads_from_campaign test due to potential API key issue.")
            exit() # Exit if API key test fails
    else:
        print("Failed to get a response from list_campaigns. Check network or API status.")
        print("Skipping list_leads_from_campaign test.")
        exit() # Exit if API key test fails


    print(f"Fetching leads from campaign ID: {test_campaign_id}")
    leads = instantly_service.list_leads_from_campaign(test_campaign_id, search)
    
    if leads:
        print(f"Total leads fetched: {len(leads)}")
        print("--- Full data for the first lead (if available) ---")
        if len(leads) > 0:
            print("Lead 1 data:")
            pprint.pprint(leads[0])
        
        if len(leads) > 1:
            print("\n--- Full data for the second lead (if available) ---")
            print("Lead 2 data:")
            pprint.pprint(leads[1])
        
        if len(leads) == 0: # This case should ideally be caught by the 'elif not leads' below
            print("No leads found in the campaign.")

    elif isinstance(leads, list) and not leads: # Check if it's an empty list (meaning no leads or error handled by function)
         print("No leads returned. This could be due to an empty campaign, an error during fetching (check logs above), or incorrect campaign ID.")
    else: # Should not happen if function returns [] on error
        print("An unexpected issue occurred while fetching leads.")