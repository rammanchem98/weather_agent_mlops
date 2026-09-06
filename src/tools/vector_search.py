import os
import logging
from google import genai
from google.genai import types
from qdrant_client import QdrantClient

from src.common.config import ROOT_DIR, get_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

config = get_config()

EMBEDDING_MODEL = config["vector_store"]["embedding_model"]
STORAGE_PATH = os.path.join(ROOT_DIR, config["vector_store"]["storage_path"])
COLLECTION_NAME = config["vector_store"]["collection_name"]

client = genai.Client()
qdrant_client = QdrantClient(path=STORAGE_PATH)


def search_air_quality_db(query: str) -> str:
    """Searches a local historical database for STORED, PAST air quality and
       weather records (temperature, humidity, PM2.5) for world cities. Use this
       whenever the user asks about historical, past, prior-year, or stored data —
       NOT for current/live/real-time conditions (use get_live_weather_api for that).

       Always attempt this search for any historical-sounding question, even if
       you are unsure whether data exists for the specific city asked about — the
       search will simply return no results if the city isn't in the database,
       which is more informative to the user than declining without checking.

       Args:
        query: The natural language search query describing what historical
               data is being requested (e.g. "historical air quality in Beijing")

    """

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(
            task_type='RETRIEVAL_QUERY',
            output_dimensionality=768
        )
    )

    query_vector = response.embeddings[0].values

    search_result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=5
    ).points

    context_doc = []
    for point in search_result:
        context_doc.append(point.payload['text'])

    return "\n".join(context_doc)
