from pathlib import Path

from azure.ai.ml import MLClient, load_component
from azure.identity import DefaultAzureCredential


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "azureml_config.json"
COMPONENT_PATH = ROOT / "azureml" / "components" / "train" / "component.yml"

credential = DefaultAzureCredential()

ml_client = MLClient.from_config(
    credential=credential,
    path=CONFIG_PATH,
)

component = load_component(source=COMPONENT_PATH)

registered_component = ml_client.components.create_or_update(component)

print("-----------------------------------------")
print("Train component registrado correctamente")
print("-----------------------------------------")
print(f"Nombre  : {registered_component.name}")
print(f"Versión : {registered_component.version}")
print(f"ID      : {registered_component.id}")