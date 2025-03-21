import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
from getpass import getpass

serpapi_params = {
    'engine': 'google',  
    'api_key': os.getenv('SERP_API_KEY') or getpass('SerpAPI key: ')  # Get the API key securely.
}

