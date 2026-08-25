import logging
import pandas as pd
from google.cloud import bigquery

from src.common.config import get_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

config = get_config()

PROJECT_ID = config["gcp"]["project_id"]
DATASET_ID = config["bigquery"]["dataset_id"]
TABLE_ID = config["bigquery"]["table_id"]
SQL_QUERY = (
    f"SELECT {config['bigquery']['columns']} "
    f"FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}` "
    f"LIMIT {config['bigquery']['row_limit']}"
)

bq_client = bigquery.Client(project=PROJECT_ID)


def fetch_air_quality_data() -> pd.DataFrame:
    """Runs the configured BigQuery extraction query and returns the raw
    air-quality/weather records as a DataFrame."""
    logging.info("Executing BigQuery data extraction query...")
    df = bq_client.query(SQL_QUERY).to_dataframe()
    logging.info(f"Fetched {len(df)} records from BigQuery.")
    return df


if __name__ == "__main__":
    print(fetch_air_quality_data().head())
