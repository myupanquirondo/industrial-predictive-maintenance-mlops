import argparse
import json
import os
from pathlib import Path

import mlflow
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


def train_model(
    df: pd.DataFrame,
    model_output: Path,
    metrics_output: Path,
) -> None:

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    print("=== AZURE ML - TRAIN ===")
    print(f"Registros: {len(df)}")
    print(f"Features: {X.shape[1]}")
    print("Distribución del target:")
    print(y.value_counts())

    # División entrenamiento / prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Peso para compensar el desbalance
    negative = (y_train == 0).sum()
    positive = (y_train == 1).sum()
    scale_pos_weight = negative / positive

    print(f"Entrenamiento: {X_train.shape}")
    print(f"Prueba: {X_test.shape}")
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")

    # Modelo
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )

    # Entrenamiento
    model.fit(X_train, y_train)

    # Predicciones
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    # Métricas
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

    # Registrar parámetros en MLflow
    mlflow.log_params(
        {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "random_state": 42,
            "scale_pos_weight": scale_pos_weight,
        }
    )

    # Registrar métricas en MLflow
    mlflow.log_metrics(metrics)

    # Crear directorios de salida
    model_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Guardar modelo XGBoost
    model_path = model_output / "model.json"
    model.save_model(model_path)

    # Guardar métricas como JSON
    metrics_path = metrics_output / "metrics.json"

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    print("\n=== RESULTADOS ===")
    print(f"PR-AUC   : {pr_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")

    print("\nModelo guardado:")
    print(model_path)

    print("\nMétricas guardadas:")
    print(metrics_path)


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_data",
        required=True,
    )

    parser.add_argument(
        "--model_output",
        required=True,
    )

    parser.add_argument(
        "--metrics_output",
        required=True,
    )

    args = parser.parse_args()

    input_path = Path(args.input_data)
    model_output = Path(args.model_output)
    metrics_output = Path(args.metrics_output)

    # Cargar datos preparados
    df = pd.read_csv(
        input_path / "prepared_data.csv"
    )

    # Azure ML proporciona AZUREML_RUN_ID cuando
    # el script se ejecuta como parte de un job.
    azureml_run_id = os.getenv("AZUREML_RUN_ID")

    if azureml_run_id:

        print(f"Azure ML Run ID: {azureml_run_id}")
        print("Usando el run administrado por Azure ML.")

        # IMPORTANTE:
        # No llamar a mlflow.set_experiment()
        # No llamar a mlflow.start_run()
        #
        # Azure ML ya administra el run.
        train_model(
            df,
            model_output,
            metrics_output,
        )

    else:

        print("Ejecución local detectada.")
        print("Creando run local de MLflow.")

        mlflow.set_experiment(
            "predictive-maintenance-local"
        )

        with mlflow.start_run():

            train_model(
                df,
                model_output,
                metrics_output,
            )


if __name__ == "__main__":
    main()