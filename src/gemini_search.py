from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from dotenv import load_dotenv
import os

# Constants
MODEL_ID = "gemini-2.0-flash"


def query_gemini_google_search(query: str):
    """Initializes client, queries Gemini with Google Search grounding, and returns only the text response.

    Args:
        query: The user's query string.

    Returns:
        response_text (str | None): The main text response from the model, or None if an error occurs
                                   during initialization or the API call.
    """
    # --- Initialize Client --- 
    client = None # Initialize client to None
    try:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("API key not found. Please set GOOGLE_API_KEY or GEMINI_API_KEY in env vars or .env file.")
        
        client = genai.Client(api_key=api_key)
        print("Gemini client initialized successfully for this query.")
    except Exception as e:
        print(f"An error occurred during client initialization: {e}")
        return None # Return error state if client fails to initialize

    # Proceed only if client initialization was successful
    if not client:
         print("Initialization failed, cannot proceed.")
         return None

    # --- Prepare and Send Query --- 
    search_tool = Tool(google_search=GoogleSearch())
    response_text = None

    print(f"\nSending query to Gemini (Model: {MODEL_ID}): \"{query}\"")
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=query,
            config=GenerateContentConfig( # Corrected parameter name
                tools=[search_tool],
                response_modalities=["TEXT"], # Ensure text modality
            )
        )

        # --- Process Response --- 
        if response.candidates and response.candidates[0].content.parts:
            response_text = response.candidates[0].content.parts[0].text
            print("Response received.")
        else:
             print("\nReceived empty or unexpected response format.")
             print(response) # Print the full response for debugging

    except Exception as e:
        print(f"\nAn error occurred during the Gemini API call: {e}")
        # Return empty/None values on error
        return None

    return response_text

def query_gemini_with_grounding(query: str):
    """Initializes client and queries Gemini model with Google Search grounding.

    Args:
        query: The user's query string.

    Returns:
        A tuple containing:
        - response_text (str | None): The main text response from the model.
        - web_queries (list[str]): List of web search queries used by the model.
        - grounding_chunks (list[dict]): List of grounding chunk details (type, title, uri).
        Returns (None, [], []) on error during initialization or API call.
    """
    # --- Initialize Client --- 
    client = None # Initialize client to None
    try:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("API key not found. Please set GOOGLE_API_KEY or GEMINI_API_KEY in env vars or .env file.")
        
        client = genai.Client(api_key=api_key)
        print("Gemini client initialized successfully for this query.")
    except Exception as e:
        print(f"An error occurred during client initialization: {e}")
        return None, [], [] # Return error state if client fails to initialize

    # Proceed only if client initialization was successful
    if not client:
         print("Initialization failed, cannot proceed.")
         return None, [], []

    # --- Prepare and Send Query --- 
    search_tool = Tool(google_search=GoogleSearch())
    response_text = None
    web_queries = []
    grounding_chunks_info = []

    print(f"\nSending query to Gemini (Model: {MODEL_ID}): \"{query}\"")
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=query,
            config=GenerateContentConfig( # Corrected parameter name
                tools=[search_tool],
                response_modalities=["TEXT"], # Ensure text modality
            )
        )

        # --- Process Response --- 
        if response.candidates and response.candidates[0].content.parts:
            response_text = response.candidates[0].content.parts[0].text
            print("Response received.")

            # Extract grounding metadata
            if hasattr(response.candidates[0], 'grounding_metadata') and response.candidates[0].grounding_metadata:
                metadata = response.candidates[0].grounding_metadata
                print("Grounding metadata found.")

                # Get Web Search Queries
                if hasattr(metadata, 'web_search_queries') and metadata.web_search_queries:
                    web_queries = list(metadata.web_search_queries)

                # Get Grounding Chunks
                if hasattr(metadata, 'grounding_chunks') and metadata.grounding_chunks:
                    for chunk in metadata.grounding_chunks:
                        chunk_info = {"type": "Unknown", "title": None, "uri": None}
                        if hasattr(chunk, 'web') and chunk.web:
                            chunk_info["type"] = "Web"
                            chunk_info["title"] = chunk.web.title
                            chunk_info["uri"] = chunk.web.uri
                        # Add elif for other chunk types if needed
                        grounding_chunks_info.append(chunk_info)
            else:
                print("No grounding metadata found in response.")
        else:
             print("\nReceived empty or unexpected response format.")
             print(response) # Print the full response for debugging

    except Exception as e:
        print(f"\nAn error occurred during the Gemini API call: {e}")
        # Return empty/None values on error
        return None, [], []

    return response_text, web_queries, grounding_chunks_info

# --- Example Usage --- 
if __name__ == "__main__":
    try:
        # Example query from previous runs
        # test_query = "When is the next total solar eclipse in the United States?"
        test_query = "Catalysta Podcast Host"

        # Call the function directly, client is initialized inside
        resp_text, queries, chunks = query_gemini_with_grounding(test_query)

        # Check if the function returned successfully before printing results
        if resp_text is not None:
            print("\n--- Function Call Results ---")
            print(f"Response Text:\n{resp_text}")

            print("\nWeb Search Queries Used:")
            if queries:
                for q in queries:
                    print(f"- {q}")
            else:
                print("(None)")

            print("\nGrounding Chunks (Sources):")
            if chunks:
                for i, c in enumerate(chunks):
                    print(f"- Chunk {i+1}: Type={c['type']}, Title='{c['title']}', URI='{c['uri']}'")
            else:
                print("(None)")
        else:
            print("\nFunction execution failed, check logs for errors.")

    except Exception as e:
        print(f"\nAn error occurred in the main execution block: {e}")

    print("\nScript finished.")
