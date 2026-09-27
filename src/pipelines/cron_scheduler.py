from google.cloud import aiplatform
from src.pipelines.ingestion_pipeline import weather_ingestion_pipeline
from src.pipelines.pipeline_config import PROJECT_ID, REGION, STAGING_BUCKET, NETWORK, SERVICE_ACCOUNT
from kfp import compiler

def create_schedule():
    compiler.Compiler().compile(weather_ingestion_pipeline, "weather_ingestion_pipeline.json")
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=STAGING_BUCKET)

    pipeline_job = aiplatform.PipelineJob(
        display_name="weather-ingestion-pipeline",
        template_path="weather_ingestion_pipeline.json",
        parameter_values={"env": "prod"},
    )
    pipeline_job.create_schedule(
        display_name="daily-weather-ingestion",
        cron="0 6 * * *",
        max_concurrent_run_count=1,
        service_account=SERVICE_ACCOUNT,
        network=NETWORK,
    )

if __name__ == "__main__":
    create_schedule()