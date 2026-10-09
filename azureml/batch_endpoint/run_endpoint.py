from pathlib import Path

from azure.ai.ml import Input, MLClient
from azure.identity import DefaultAzureCredential


ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT / "configs" / "azureml_config.json"
INPUT_PATH = ROOT / "data" / "test_batch" / "input.csv"


credential = DefaultAzureCredential()

ml_client = MLClient.from_config(
    credential=credential,
    path=CONFIG_PATH,
)


print("-----------------------------------------")
print("Ejecutando Batch Endpoint...")
print("-----------------------------------------")


job = ml_client.batch_endpoints.invoke(
    endpoint_name="predictive-maintenance-batch",
    input=Input(
        type="uri_file",
        path=str(INPUT_PATH),
    ),
)


print("-----------------------------------------")
print("Batch job enviado correctamente")
print("-----------------------------------------")
print(f"Job ID: {job.name}")