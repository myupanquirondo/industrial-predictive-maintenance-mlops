from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "azureml_config.json"
CONDA_FILE = ROOT / "azureml" / "environments" / "training" / "conda.yml"


credential = DefaultAzureCredential()

ml_client = MLClient.from_config(
    credential=credential,
    path=CONFIG_PATH,
)


environment = Environment(
    name="predictive-maintenance-training",
    description="Environment for XGBoost training and MLflow tracking.",
    image="mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04",
    conda_file=str(CONDA_FILE),
)


registered_environment = ml_client.environments.create_or_update(
    environment
)


print("-----------------------------------------")
print("Environment registrado correctamente")
print("-----------------------------------------")
print(f"Nombre  : {registered_environment.name}")
print(f"Versión : {registered_environment.version}")
print(f"ID      : {registered_environment.id}")