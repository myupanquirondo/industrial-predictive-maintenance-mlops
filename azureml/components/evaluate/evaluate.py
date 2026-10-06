import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


TARGET = "Machine failure"


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_input",
        required=True,
    )

    parser.add_argument(
        "--data_input",
        required=True,
    )

    parser.add_argument(
        "--metrics_output",
        required=True,
    )

    args = parser.parse_args()

    model_input = Path(args.model_input)
    data_input = Path(args.data_input)
    metrics_output = Path(args.metrics_output)

    # Cargar datos preparados
    data_file = data_input / "prepared_data.csv"
    df = pd.read_csv(data_file)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    print("=== AZURE ML - EVALUATE ===")
    print(f"Registros totales: {len(df)}")

    # Reproducir la misma división utilizada durante el entrenamiento.
    # El modelo fue entrenado con el 80 % y se evaluará sobre el 20 % restante.
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Registros de evaluación: {len(X_test)}")

    # Cargar modelo entrenado
    model = XGBClassifier()
    model.load_model(
        model_input / "model.json"
    )

    # Generar predicciones únicamente sobre el conjunto de prueba
    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # Calcular métricas
    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    metrics = {
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    # Crear directorio de salida
    metrics_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Guardar métricas
    metrics_file = (
        metrics_output / "evaluation_metrics.json"
    )

    with open(
        metrics_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    print("\n=== RESULTADOS DE EVALUACIÓN ===")
    print(f"PR-AUC   : {pr_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")

    print("\nMétricas guardadas en:")
    print(metrics_file)


if __name__ == "__main__":
    main()