import argparse
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


FEATURES = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

TARGET = "Machine failure"


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_data", required=True)
    parser.add_argument("--output_data", required=True)

    args = parser.parse_args()

    input_path = Path(args.input_data)
    output_path = Path(args.output_data)

    # Azure ML entrega uri_folder como una carpeta.
    csv_files = sorted(input_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No se encontraron archivos CSV en: {input_path}"
        )

    print("=== AZURE ML - PREPARE ===")
    print(f"Archivos encontrados: {len(csv_files)}")

    # Leer y combinar los batches.
    dataframes = [
        pd.read_csv(csv_file)
        for csv_file in csv_files
    ]

    df = pd.concat(
        dataframes,
        ignore_index=True,
    )

    print(f"Registros totales: {len(df)}")

    # Seleccionar variables predictoras y target.
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # One-Hot Encoding de Type.
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    encoded_type = encoder.fit_transform(
        X[["Type"]]
    )

    encoded_columns = encoder.get_feature_names_out(
        ["Type"]
    )

    X_encoded = X[
        [
            "Air temperature",
            "Process temperature",
            "Rotational speed",
            "Torque",
            "Tool wear",
        ]
    ].copy()

    for index, column in enumerate(encoded_columns):
        X_encoded[column] = encoded_type[:, index]

    # Conservar el target para TRAIN.
    X_encoded[TARGET] = y.values

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = output_path / "prepared_data.csv"

    X_encoded.to_csv(
        output_file,
        index=False,
    )

    print(f"Features: {X_encoded.shape[1] - 1}")
    print(f"Target: {TARGET}")
    print(f"Salida: {output_file}")


if __name__ == "__main__":
    main()