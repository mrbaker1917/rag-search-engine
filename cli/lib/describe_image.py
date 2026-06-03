import os
from urllib import response
import mimetypes
from dotenv import load_dotenv
from google import genai
from google.genai import types

from .hybrid_search import HybridSearch
from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    RRF_K,
    SEARCH_MULTIPLIER,
    load_movies,
)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)
model = "gemma-4-31b-it"

def describe_command(image_path, query):
    if not image_path:
        print("Please provide an image path using --image")
        return
    
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    prompt = f"""Given the included image and text query, rewrite the text query to improve search 
    results from a movie database. Make sure to:
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary"""
    
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"
    
    parts = [
    prompt,
    types.Part.from_bytes(data=image_bytes, mime_type=mime),
    query.strip(),
]
    response = client.models.generate_content(model=model, contents=parts)
    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")