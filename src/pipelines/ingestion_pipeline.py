from kfp import dsl, compiler

from src.pipelines.pipeline_config import PROJECT_ID

PIPELINE_IMAGE = "europe-west2-docker.pkg.dev/project-f0ff9de1-dfa0-4dca-957/weather-agent-repo/pipeline-image:latest"


@dsl.component(base_image=PIPELINE_IMAGE, packages_to_install=["google-cloud-secret-manager"])
def ingest_live_weather_op(env: str, openweather_secret_id: str, project_id: str):
    import os
    from google.cloud import secretmanager

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{openweather_secret_id}/versions/latest"
    api_key = client.access_secret_version(request={"name": name}).payload.data.decode("UTF-8")

    os.environ["env"] = env
    os.environ["OPENWEATHER_API_KEY"] = api_key
    from src.data.ingest_live_weather import fetch_and_ingest_data
    fetch_and_ingest_data()


@dsl.component(base_image=PIPELINE_IMAGE, packages_to_install=["google-cloud-secret-manager"])
def index_qdrant_op(env: str, gemini_secret_id: str, project_id: str):
    import os
    from google.cloud import secretmanager

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{gemini_secret_id}/versions/latest"
    api_key = client.access_secret_version(request={"name": name}).payload.data.decode("UTF-8")

    os.environ["env"] = env
    os.environ["GEMINI_API_KEY"] = api_key
    from src.data.index_qdrant import run_ingestion
    run_ingestion()


@dsl.pipeline(name="weather-ingestion-pipeline")
def weather_ingestion_pipeline(
    env: str = "prod",
    openweather_secret_id: str = "openweather-api-key",
    gemini_secret_id: str = "gemini-api-key",
    project_id: str = PROJECT_ID,
):
    ingest_step = ingest_live_weather_op(env=env, openweather_secret_id=openweather_secret_id, project_id=project_id)
    index_step = index_qdrant_op(env=env, gemini_secret_id=gemini_secret_id, project_id=project_id).after(ingest_step)


if __name__ == "__main__":
 
    compiler.Compiler().compile(weather_ingestion_pipeline, "weather_ingestion_pipeline.json")

# .after() enforces order without a data-artifact dependency —
# these two communicate via BigQuery/Qdrant side effects, not KFP artifacts