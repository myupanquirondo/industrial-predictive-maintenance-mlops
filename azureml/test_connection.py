from pathlib import Path

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "azureml_config.json"

credential = DefaultAzureCredential()

ml_client = MLClient.from_config(
    credential=credential,
    path=CONFIG_PATH,
)

workspace = ml_client.workspaces.get(ml_client.workspace_name)

print("-----------------------------------------")
print("Azure ML conectado correctamente")
print("-----------------------------------------")
print(f"Workspace : {workspace.name}")
print(f"Región    : {workspace.location}")
print(f"Resource Group : {workspace.resource_group}")