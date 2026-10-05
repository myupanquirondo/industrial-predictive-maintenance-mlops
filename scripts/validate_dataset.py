from ucimlrepo import fetch_ucirepo


def main() -> None:
    dataset = fetch_ucirepo(id=601)

    features = dataset.data.features
    targets = dataset.data.targets

    print("=== DATASET ===")
    print(f"Filas: {len(features)}")
    print(f"Features: {features.shape[1]}")
    print(f"Targets: {targets.shape[1]}")

    print("\n=== COLUMNAS FEATURES ===")
    print(features.columns.tolist())

    print("\n=== TIPOS DE DATOS ===")
    print(features.dtypes)

    print("\n=== VALORES NULOS ===")
    print(features.isnull().sum())

    print("\n=== DUPLICADOS ===")
    print(f"Duplicados: {features.duplicated().sum()}")

    print("\n=== DISTRIBUCIÓN DE MACHINE FAILURE ===")
    print(targets["Machine failure"].value_counts())
    print(targets["Machine failure"].value_counts(normalize=True))


if __name__ == "__main__":
    main()