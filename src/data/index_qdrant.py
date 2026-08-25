import os
import time
import logging
from google import genai
from google.genai import types, errors
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.common.config import ROOT_DIR, get_config
from src.data.fetch_raw_data import fetch_air_quality_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

config = get_config()

COLLECTION_NAME = config["vector_store"]["collection_name"]
STORAGE_PATH = os.path.join(ROOT_DIR, config["vector_store"]["storage_path"])
EMBEDDING_MODEL = config["vector_store"]["embedding_model"]

qdrant_client = QdrantClient(path=STORAGE_PATH)
genai_client = genai.Client()


def init_collection():
    """Recreates the collection from scratch so each ingestion run is a full
    refresh — otherwise stale points from a previous run/dataset would
    linger (upserts only overwrite matching IDs, never remove extras)."""
    logging.info(f"Checking if Qdrant collection '{COLLECTION_NAME}' exists...")
    collections_response = qdrant_client.get_collections()
    existing_names = [col.name for col in collections_response.collections]

    if COLLECTION_NAME in existing_names:
        logging.info(f"Collection '{COLLECTION_NAME}' already exists. Deleting for a full refresh...")
        qdrant_client.delete_collection(collection_name=COLLECTION_NAME)

    logging.info(f"Creating collection '{COLLECTION_NAME}'...")
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )
    logging.info("Collection created successfully.")


def embed_with_retry(contents, embed_config, max_retries=5, base_delay=5):
    """Retries only on 429 (rate limit) with exponential backoff. Other
    errors (e.g. a genuinely exhausted daily quota) are raised immediately
    since waiting won't help."""
    for attempt in range(max_retries):
        try:
            return genai_client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=contents,
                config=embed_config
            )
        except errors.APIError as e:
            if e.code == 429 and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logging.warning(f"Rate limited (429). Retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise


def run_ingestion():
    init_collection()

    df = fetch_air_quality_data()

    def row_format(row):
        return f"city:{row['city']},temp:{row['temp']},humidity:{row['humidity']},wind_Speed:{row['wind_speed']},pm25_value:{row['pm25_value']}"

    documents = df.apply(row_format, axis=1).tolist()
    list_of_embeddings = []
    batch_size = 20

    for i in range(0, len(documents), batch_size):
        chunk_docs = documents[i:i+batch_size]
        logging.info(f"Processing batch size: {len(chunk_docs)}")

        response = embed_with_retry(
            chunk_docs,
            types.EmbedContentConfig(
                task_type='RETRIEVAL_DOCUMENT',
                output_dimensionality=768
            )
        )
        time.sleep(4)
        list_of_embeddings.extend([e.values for e in response.embeddings])

    if len(list_of_embeddings) != len(documents):
        raise ValueError("Length mismatch between embeddings and documents.")

    qdrant_points = []
    for idx, (doc, vector) in enumerate(zip(documents, list_of_embeddings)):
        row_metadata = df.iloc[idx]
        point_id = idx + 1

        point = PointStruct(
            id=point_id,
            vector=vector,
            payload={
                'city': row_metadata['city'],
                'text': doc
            }
        )
        qdrant_points.append(point)

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=qdrant_points
    )

    logging.info("Vector ingestion completed successfully.")


if __name__ == "__main__":
    run_ingestion()
