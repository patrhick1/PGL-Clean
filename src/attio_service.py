import requests
import os
import time
import logging
from typing import Dict, Any, Optional, List, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AttioClient:
    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3, retry_delay: int = 2):
        self.api_key = api_key or os.getenv("ATTIO_ACCESS_TOKEN")
        if not self.api_key:
            raise ValueError("Attio API key not found. Please set ATTIO_ACCESS_TOKEN environment variable or pass api_key parameter.")
        
        self.base_url = "https://api.attio.com/v2"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def create_record(self, object_type: str, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a record in Attio for any object type
        
        Args:
            object_type: The type/slug of the object (e.g., 'podcast', 'company')
            attributes: Dictionary of attribute values to set
            
        Returns:
            API response as dictionary
            
        Raises:
            ValueError: If required parameters are missing
            requests.exceptions.RequestException: For API and network errors
        """
        # Validate inputs
        if not object_type:
            raise ValueError("Object type is required")
        if not attributes:
            raise ValueError("At least one attribute value is required")
        
        # Prepare payload
        payload = {
            "data": {
                "values": attributes
            }
        }
        
        # Construct endpoint and make request
        endpoint = f"{self.base_url}/objects/{object_type}/records"
        return self._make_api_request("post", endpoint, payload)
    
    def get_record(self, object_type: str, record_id: str) -> Dict[str, Any]:
        """
        Get a record by its ID
        
        Args:
            object_type: The type/slug of the object (e.g., 'podcast', 'company')
            record_id: The ID of the record to retrieve
            
        Returns:
            API response as dictionary
        """
        endpoint = f"{self.base_url}/objects/{object_type}/records/{record_id}"
        return self._make_api_request("get", endpoint)
    
    def update_record(self, object_type: str, record_id: str, attributes: Dict[str, Any], overwrite: bool = False) -> Dict[str, Any]:
        """
        Update a record in Attio
        
        Args:
            object_type: The type/slug of the object (e.g., 'podcast', 'company')
            record_id: The ID of the record to update
            attributes: Dictionary of attribute values to update
            overwrite: If True, use PUT to overwrite multiselect values, otherwise use PATCH to append values
            
        Returns:
            API response as dictionary
        """
        # Format the payload
        payload = {
            "data": {
                "values": attributes
            }
        }
        
        # Determine the endpoint and method
        endpoint = f"{self.base_url}/objects/{object_type}/records/{record_id}"
        method = "put" if overwrite else "patch"
        
        # Make the request
        return self._make_api_request(method, endpoint, payload)
    
    def delete_record(self, object_type: str, record_id: str) -> Dict[str, Any]:
        """
        Delete a record by its ID
        
        Args:
            object_type: The type/slug of the object (e.g., 'podcast', 'company')
            record_id: The ID of the record to delete
            
        Returns:
            API response as dictionary
        """
        endpoint = f"{self.base_url}/objects/{object_type}/records/{record_id}"
        return self._make_api_request("delete", endpoint)
    
    def list_records(self, object_type: str, filters: Optional[Dict[str, Any]] = None, 
                    page: int = 1, limit: int = 100) -> Dict[str, Any]:
        """
        List records of a specific object type, with optional filtering
        
        Args:
            object_type: The type/slug of the object (e.g., 'podcast', 'company')
            filters: Optional dictionary of filters to apply
            page: Page number for pagination
            limit: Number of records per page
            
        Returns:
            API response as dictionary
        """
        # Prepare payload for POST /query endpoint
        payload = {
            "pagination": {
                "page": page,
                "limit": limit
            }
        }
        
        # Add filters if provided
        if filters:
            payload["filter"] = filters
        
        # Construct endpoint and make request
        endpoint = f"{self.base_url}/objects/{object_type}/records/query"
        return self._make_api_request("post", endpoint, payload)
    
    def _make_api_request(self, method: str, url: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make API request with retry logic and error handling"""
        retries = 0
        while retries <= self.max_retries:
            try:
                logger.info(f"Making {method.upper()} request to {url}")
                if data:
                    logger.info(f"Payload: {data}")
                
                if method.lower() == "post":
                    response = requests.post(url, json=data, headers=self.headers, timeout=10)
                elif method.lower() == "put":
                    response = requests.put(url, json=data, headers=self.headers, timeout=10)
                elif method.lower() == "patch":
                    response = requests.patch(url, json=data, headers=self.headers, timeout=10)
                elif method.lower() == "delete":
                    response = requests.delete(url, headers=self.headers, timeout=10)
                else:  # Default to GET
                    response = requests.get(url, headers=self.headers, timeout=10)
                
                # Log response for debugging
                try:
                    response_data = response.json()
                    logger.info(f"Response status: {response.status_code}")
                    logger.info(f"Response body: {response_data}")
                except ValueError:
                    logger.info(f"Response status: {response.status_code}")
                    logger.info(f"Response text: {response.text}")
                
                # Handle rate limiting (status code 429)
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', self.retry_delay))
                    logger.warning(f"Rate limited. Retrying after {retry_after} seconds.")
                    time.sleep(retry_after)
                    retries += 1
                    continue
                
                # Raise exception for client and server errors
                response.raise_for_status()
                
                # Return successful response
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {str(e)}")
                if retries >= self.max_retries:
                    logger.error("Max retries reached. Raising exception.")
                    raise
                
                logger.info(f"Retrying in {self.retry_delay} seconds... (Attempt {retries + 1}/{self.max_retries})")
                time.sleep(self.retry_delay)
                retries += 1

    def filter_records(self, object_type: str, attribute_name: str, value: Any, 
                  operator: str = "equals", page: int = 1, limit: int = 100) -> Dict[str, Any]:
        """
        Filter records where a specific attribute has a particular value, using a simple equality check.
        This method is a convenience wrapper for the common case of filtering for an exact match
        as shown in Attio's API documentation (e.g., {"name": "Ada Lovelace"}).

        For more complex filtering logic (e.g., using operators other than equals, $and/$or combinations),
        use the `query_records` method directly, providing the full Attio filter object.
        
        Args:
            object_type: The type/slug of the object (e.g., 'people').
            attribute_name: The slug of the attribute to filter on (e.g., 'email_addresses', 'name').
            value: The value to filter by. 
                   If filtering an attribute that expects a single scalar (e.g. a name), provide that scalar.
                   If filtering an array attribute (like 'email_addresses') for an exact match of one of its items,
                   provide that single item (e.g., 'test@example.com', not ['test@example.com']).
                   The method currently handles if a single-item list is passed for `value` by using its first element.
            operator: The comparison operator. Only "equals" (case-insensitive) is supported by this method.
            page: Page number for pagination.
            limit: Number of records per page.
            
        Returns:
            API response as dictionary.

        Raises:
            NotImplementedError: If an operator other than "equals" is specified.
        """
        query_filter = None
        if operator.lower() == "equals":
            # Attio's simple filter example is { "attribute_slug": direct_value }
            # If the provided value is a list with one item (e.g., from previous usage patterns for email_addresses),
            # extract the item to match the simple string filter that worked.
            actual_value = value
            if isinstance(value, list) and len(value) == 1:
                actual_value = value[0]
                logger.info(f"Simplified filter: Using first element of list for attribute '{attribute_name}'. Value: '{actual_value}'")
            elif isinstance(value, list) and len(value) != 1:
                # This case is ambiguous for a simple equality filter like {"slug": "value"}
                logger.warning(
                    f"Filtering attribute '{attribute_name}' with operator 'equals' and a list value with multiple items or empty: {value}. "
                    f"This may not behave as expected with Attio's simple filter. Consider using query_records for complex array matching."
                    f"Proceeding with the list as is, but direct scalar or single-item list is preferred for this method."
                )
                # query_filter = {attribute_name: value} # Let it pass as is, Attio will decide.

            query_filter = {attribute_name: actual_value}
        else:
            raise NotImplementedError(
                f"Operator '{operator}' is not supported by filter_records. "
                f"This method only supports 'equals'. For other operators or complex filters, use query_records."
            )

        return self.list_records(object_type, query_filter, page, limit)

    def query_records(self, object_type: str, filters: Dict[str, Any], page: int = 1, limit: int = 100) -> Dict[str, Any]:
        """
        Query records of a specific object type with full control over Attio's filter syntax.
        
        Args:
            object_type: The type/slug of the object (e.g., 'people', 'companies').
            filters: Dictionary representing the Attio filter object (e.g., {"$and": [...], "status": "Open"}).
                     Refer to Attio API documentation for filter structure.
            page: Page number for pagination.
            limit: Number of records per page.
            
        Returns:
            API response as dictionary.
        """
        # Prepare payload for POST /query endpoint
        payload = {
            "pagination": {
                "page": page,
                "limit": limit
            }
        }
        
        # Add filters if provided
        if filters:
            payload["filter"] = filters
        
        # Construct endpoint and make request
        endpoint = f"{self.base_url}/objects/{object_type}/records/query"
        return self._make_api_request("post", endpoint, payload)

    def create_note(self, parent_object_slug: str, parent_record_id: str, title: str, content: str, created_at: str, note_format: str = "plaintext"):
        """
        Creates a new note for a given record in Attio.

        Args:
            parent_object_slug: The slug of the parent object (e.g., "people").
            parent_record_id: The ID of the parent record.
            title: The title of the note.
            content: The content of the note.
            created_at: The ISO 8601 timestamp for when the note was created.
            note_format: The format of the note content ("plaintext" or "markdown").

        Returns:
            The JSON response from the Attio API.

        Raises:
            requests.exceptions.HTTPError: For HTTP errors from the Attio API.
            requests.exceptions.RequestException: For other request-related errors.
        """
        endpoint = f"{self.base_url}/notes"
        payload = {
            "data": {
                "parent_object": parent_object_slug,
                "parent_record_id": parent_record_id,
                "title": title,
                "format": note_format,
                "content": content,
            }
        }
        # Only add created_at if it's provided and valid
        if created_at:
            payload["data"]["created_at"] = created_at
        
        logger.info(f"Creating note for record {parent_record_id} in object {parent_object_slug} with title '{title}'")
        try:
            # _make_api_request already handles raise_for_status and returns the parsed JSON dictionary on success
            json_response = self._make_api_request("post", endpoint, data=payload)
            
            # Log success using the json_response directly
            note_id = json_response.get('data', {}).get('id', {}).get('note_id')
            logger.info(f"Successfully created note for record {parent_record_id}. Note ID: {note_id}")
            return json_response # Return the parsed JSON dictionary
        except requests.exceptions.HTTPError as http_err:
            # _make_api_request should raise this if there was an HTTP error after retries
            logger.error(f"HTTP error creating note for {parent_record_id} in {parent_object_slug} (already handled by _make_api_request but caught again): {http_err} - {http_err.response.text if http_err.response else 'No response body'}")
            raise # Re-raise it to be caught by the main script
        except requests.exceptions.RequestException as req_err:
            # _make_api_request should raise this if there was a request error after retries
            logger.error(f"Request error creating note for {parent_record_id} in {parent_object_slug} (already handled by _make_api_request but caught again): {req_err}")
            raise # Re-raise it
        except Exception as e:
            logger.error(f"Unexpected error in create_note for {parent_record_id} in {parent_object_slug} after API call: {e}")
            raise

# Example usage
if __name__ == "__main__":
    attio = AttioClient()
    filters = {
        "host_name": "Max"  # Simple equals filter
    }
    podcast_records = attio.query_records("podcast", filters)
    print(podcast_records)
        
        # Create a podcast record
