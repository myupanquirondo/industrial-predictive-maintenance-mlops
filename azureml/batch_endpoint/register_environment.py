from pathlib import Path

from azure.ai.ml import MLClient, load_environment
from azure.identity import DefaultAzureCredential


ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT / "configs" / "azureml_config.json"

ENVIRONMENT_PATH = (
    ROOT
    / "azureml"
    / "batch_endpoint"
    / "environment.yml"
)


credential = DefaultAzureCredential()

ml_client = MLClient.from_config(
    credential=credential,
    path=CONFIG_PATH,
)

environment = load_environment(
    source=ENVIRONMENT_PATH
)

registered_environment = (
    ml_client.environments.create_or_update(
        environment
    )
)

print("-----------------------------------------")
print("Environment registrado correctamente")
print("-----------------------------------------")
print(f"Nombre  : {registered_environment.name}")
print(f"Versión : {registered_environment.version}")
print(f"ID      : {registered_environment.id}")