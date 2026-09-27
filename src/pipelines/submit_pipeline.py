from google.cloud import aiplatform
from kfp import compiler

from src.pipelines.ingestion_pipeline import weather_ingestion_pipeline
from src.pipelines.pipeline_config import PROJECT_ID, REGION, STAGING_BUCKET, NETWORK, SERVICE_ACCOUNT

PIPELINE_JSON = "weather_ingestion_pipeline.json"


def submit():
    compiler.Compiler().compile(weather_ingestion_pipeline, PIPELINE_JSON)
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=STAGING_BUCKET)

    job = aiplatform.PipelineJob(
        display_name="weather-ingestion-pipeline",
        template_path=PIPELINE_JSON,
        parameter_values={"env": "prod"},
    )
    job.run(sync=True, network=NETWORK, service_account=SERVICE_ACCOUNT)


if __name__ == "__main__":
    submit()