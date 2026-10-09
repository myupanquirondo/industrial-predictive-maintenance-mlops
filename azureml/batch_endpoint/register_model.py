from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential


ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT / "configs" / "azureml_config.json"
MODEL_PATH = ROOT / "data" / "test_train" / "model" / "model.json"


credential = DefaultAzureCredential()

ml_client = MLClient.from_config(
    credential=credential,
    path=CONFIG_PATH,
)


model = Model(
    name="predictive-maintenance-xgboost",
    path=str(MODEL_PATH),
    type=AssetTypes.CUSTOM_MODEL,
    description=(
        "XGBoost model for industrial predictive maintenance "
        "trained with the AI4I 2020 dataset."
    ),
)


registered_model = ml_client.models.create_or_update(model)


print("-----------------------------------------")
print("Modelo registrado correctamente")
print("-----------------------------------------")
print(f"Nombre  : {registered_model.name}")
print(f"Versión : {registered_model.version}")
print(f"ID      : {registered_model.id}")