from pathlib import Path

from azure.ai.ml import Input, MLClient, dsl, load_component
from azure.identity import DefaultAzureCredential


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "azureml_config.json"

PREPARE_COMPONENT = (
    ROOT / "azureml" / "components" / "prepare" / "component.yml"
)

TRAIN_COMPONENT = (
    ROOT / "azureml" / "components" / "train" / "component.yml"
)

EVALUATE_COMPONENT = (
    ROOT / "azureml" / "components" / "evaluate" / "component.yml"
)

DATA_PATH = ROOT / "data" / "raw" / "batches"


credential = DefaultAzureCredential()

ml_client = MLClient.from_config(
    credential=credential,
    path=CONFIG_PATH,
)


prepare = load_component(
    source=PREPARE_COMPONENT,
)

train = load_component(
    source=TRAIN_COMPONENT,
)

evaluate = load_component(
    source=EVALUATE_COMPONENT,
)


@dsl.pipeline(
    name="predictive-maintenance-pipeline",
    description=(
        "Pipeline reproducible de preparación, "
        "entrenamiento y evaluación."
    ),
)
def predictive_maintenance_pipeline(input_data):

    prepare_job = prepare(
        input_data=input_data,
    )

    train_job = train(
        input_data=prepare_job.outputs.output_data,
    )

    evaluate_job = evaluate(
        model_input=train_job.outputs.model_output,
        data_input=prepare_job.outputs.output_data,
    )

    return {
        "model": train_job.outputs.model_output,
        "metrics": evaluate_job.outputs.metrics_output,
    }


if __name__ == "__main__":

    pipeline_job = predictive_maintenance_pipeline(
        input_data=Input(
            type="uri_folder",
            path=str(DATA_PATH),
        )
    )

    pipeline_job.settings.default_compute = "serverless"

    returned_job = ml_client.jobs.create_or_update(
        pipeline_job,
        experiment_name="predictive-maintenance-pipeline",
    )

    print("-----------------------------------------")
    print("Pipeline enviado correctamente")
    print("-----------------------------------------")
    print(f"Job ID: {returned_job.name}")
    print(f"Studio: {returned_job.studio_url}")