from ucimlrepo import fetch_ucirepo


TARGET = "Machine failure"

FEATURES = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

EXCLUDED_TARGET_COMPONENTS = [
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def main() -> None:
    dataset = fetch_ucirepo(id=601)

    available_features = dataset.data.features.columns.tolist()
    available_targets = dataset.data.targets.columns.tolist()

    print("=== CONFIGURACIÓN DEL PROBLEMA ===")
    print(f"Target: {TARGET}")

    print("\n=== FEATURES ===")
    for feature in FEATURES:
        print(f"- {feature}")

    print("\n=== VARIABLES EXCLUIDAS ===")
    for column in EXCLUDED_TARGET_COMPONENTS:
        print(f"- {column}")

    print("\n=== VALIDACIÓN ===")

    for feature in FEATURES:
        if feature not in available_features:
            raise ValueError(f"Feature no encontrada: {feature}")

    if TARGET not in available_targets:
        raise ValueError(f"Target no encontrado: {TARGET}")

    for column in EXCLUDED_TARGET_COMPONENTS:
        if column not in available_targets:
            raise ValueError(f"Variable excluida no encontrada: {column}")

    print("Configuración válida.")


if __name__ == "__main__":
    main()