from pathlib import Path

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "azureml_config.json"

credential = DefaultAzureCredential()

ml_client = MLClient.from_config(
    credential=credential,
    path=CONFIG_PATH,
)

endpoint = ml_client.batch_endpoints.get(
    "predictive-maintenance-batch"
)

endpoint.defaults.deployment_name = "blue"

updated_endpoint = (
    ml_client.batch_endpoints.begin_create_or_update(endpoint)
    .result()
)

print("-----------------------------------------")
print("Deployment predeterminado configurado")
print("-----------------------------------------")
print(f"Endpoint   : {updated_endpoint.name}")
print(
    f"Deployment : "
    f"{updated_endpoint.defaults.deployment_name}"
)