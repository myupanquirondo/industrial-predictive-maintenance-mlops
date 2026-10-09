from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier


FEATURES = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
    "Type_H",
    "Type_L",
    "Type_M",
]

model = None


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_columns = {
        "Air temperature [K]": "Air temperature",
        "Process temperature [K]": "Process temperature",
        "Rotational speed [rpm]": "Rotational speed",
        "Torque [Nm]": "Torque",
        "Tool wear [min]": "Tool wear",
    }

    df = df.rename(columns=rename_columns)

    if "Type" not in df.columns:
        raise ValueError(
            "La columna 'Type' es obligatoria. "
            "Debe contener valores H, L o M."
        )

    type_dummies = pd.get_dummies(
        df["Type"],
        prefix="Type",
        dtype=int,
    )

    for column in ["Type_H", "Type_L", "Type_M"]:
        if column not in type_dummies.columns:
            type_dummies[column] = 0

    type_dummies = type_dummies[
        ["Type_H", "Type_L", "Type_M"]
    ]

    df = pd.concat(
        [
            df.drop(columns=["Type"]),
            type_dummies,
        ],
        axis=1,
    )

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Faltan las siguientes variables requeridas: "
            + ", ".join(missing_features)
        )

    return df[FEATURES]


def init() -> None:
    global model

    model_dir = Path("/var/azureml-app/azureml-models")

    model_files = list(model_dir.rglob("model.json"))

    if not model_files:
        raise FileNotFoundError(
            "No se encontró model.json dentro de "
            "/var/azureml-app/azureml-models."
        )

    model_path = model_files[0]

    print("=== BATCH ENDPOINT - INIT ===")
    print(f"Cargando modelo: {model_path}")

    model = XGBClassifier()
    model.load_model(model_path)

    print("Modelo cargado correctamente.")


def run(mini_batch):
    if model is None:
        raise RuntimeError(
            "El modelo no fue inicializado correctamente."
        )

    print("=== BATCH ENDPOINT - RUN ===")
    print(f"Archivos recibidos: {len(mini_batch)}")

    results = []

    for input_file in mini_batch:
        input_path = Path(input_file)

        print(f"Procesando: {input_path}")

        df = pd.read_csv(input_path)

        print(f"Registros: {len(df)}")

        X = prepare_features(df)

        print(f"Features utilizadas: {list(X.columns)}")

        probabilities = model.predict_proba(X)[:, 1]

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        output = df.copy()

        output["failure_probability"] = probabilities
        output["prediction"] = predictions

        output["maintenance_alert"] = output[
            "prediction"
        ].map(
            {
                0: "NORMAL",
                1: "MAINTENANCE_REQUIRED",
            }
        )

        results.append(output)

    if not results:
        raise ValueError(
            "No se recibieron archivos para procesar."
        )

    final_results = pd.concat(
        results,
        ignore_index=True,
    )

    print("=== RESULTADOS ===")
    print(
        f"Registros procesados: {len(final_results)}"
    )
    print(
        f"Fallas predichas: "
        f"{final_results['prediction'].sum()}"
    )
    print(
        f"Probabilidad máxima: "
        f"{final_results['failure_probability'].max():.4f}"
    )

    return final_results