from ucimlrepo import fetch_ucirepo
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
    dataset = fetch_ucirepo(id=601)

    X = dataset.data.features[FEATURES].copy()
    y = dataset.data.targets[TARGET].copy()

    categorical_features = ["Type"]
    numerical_features = [
        "Air temperature",
        "Process temperature",
        "Rotational speed",
        "Torque",
        "Tool wear",
    ]

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    encoded_type = encoder.fit_transform(X[categorical_features])

    encoded_columns = encoder.get_feature_names_out(categorical_features)

    X_encoded = X[numerical_features].copy()

    for index, column in enumerate(encoded_columns):
        X_encoded[column] = encoded_type[:, index]

    print("=== TRANSFORMACIÓN ===")
    print(f"Filas: {len(X_encoded)}")
    print(f"Features originales: {len(FEATURES)}")
    print(f"Features transformadas: {X_encoded.shape[1]}")

    print("\n=== COLUMNAS TRANSFORMADAS ===")
    print(X_encoded.columns.tolist())

    print("\n=== PRIMERAS FILAS ===")
    print(X_encoded.head())

    print("\n=== TARGET ===")
    print(y.value_counts())


if __name__ == "__main__":
    main()