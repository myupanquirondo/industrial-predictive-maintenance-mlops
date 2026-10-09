from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential


ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT / "configs" / "azureml_config.json"

COMPUTE_NAME = "batch-cluster"


credential = DefaultAzureCredential()

ml_client = MLClient.from_config(
    credential=credential,
    path=CONFIG_PATH,
)


existing = list(ml_client.compute.list())

if any(compute.name == COMPUTE_NAME for compute in existing):
    print("-----------------------------------------")
    print("El compute ya existe")
    print("-----------------------------------------")
    print(f"Nombre: {COMPUTE_NAME}")

else:
    compute = AmlCompute(
        name=COMPUTE_NAME,
        type="amlcompute",
        size="STANDARD_DS3_v2",
        min_instances=0,
        max_instances=1,
        idle_time_before_scale_down=120,
        description="Compute cluster for predictive maintenance batch inference",
    )

    print("Creando compute...")

    created = ml_client.begin_create_or_update(
        compute
    ).result()

    print("-----------------------------------------")
    print("Compute creado correctamente")
    print("-----------------------------------------")
    print(f"Nombre : {created.name}")
    print(f"Estado : {created.provisioning_state}")