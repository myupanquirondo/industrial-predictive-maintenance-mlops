from pathlib import Path

from azure.ai.ml import MLClient, load_batch_deployment
from azure.identity import DefaultAzureCredential


ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT / "configs" / "azureml_config.json"

DEPLOYMENT_PATH = (
    ROOT
    / "azureml"
    / "batch_endpoint"
    / "deployment.yml"
)


credential = DefaultAzureCredential()

ml_client = MLClient.from_config(
    credential=credential,
    path=CONFIG_PATH,
)


deployment = load_batch_deployment(
    source=DEPLOYMENT_PATH
)


created_deployment = ml_client.batch_deployments.begin_create_or_update(
    deployment
).result()


print("-----------------------------------------")
print("Batch deployment creado correctamente")
print("-----------------------------------------")
print(f"Nombre   : {created_deployment.name}")
print(f"Endpoint : {created_deployment.endpoint_name}")
print(f"Estado   : {created_deployment.provisioning_state}")