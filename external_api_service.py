# app/external_api_service.py

import os
from dotenv import load_dotenv
import requests

load_dotenv()

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