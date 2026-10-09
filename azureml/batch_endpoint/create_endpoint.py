from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import BatchEndpoint
from azure.identity import DefaultAzureCredential


ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT / "configs" / "azureml_config.json"

credential = DefaultAzureCredential()

ml_client = MLClient.from_config(
    credential=credential,
    path=CONFIG_PATH,
)

endpoint = BatchEndpoint(
    name="predictive-maintenance-batch",
    description="Batch endpoint for predictive maintenance inference.",
    auth_mode="aad_token",
)

print("-----------------------------------------")
print("Creando Batch Endpoint...")
print("-----------------------------------------")

created_endpoint = (
    ml_client.batch_endpoints.begin_create_or_update(endpoint)
    .result()
)

print("-----------------------------------------")
print("Batch Endpoint creado correctamente")
print("-----------------------------------------")
print(f"Nombre      : {created_endpoint.name}")
print(f"Estado      : {created_endpoint.provisioning_state}")
print(f"Scoring URI : {created_endpoint.scoring_uri}")