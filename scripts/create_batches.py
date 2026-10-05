from pathlib import Path

from ucimlrepo import fetch_ucirepo


BATCH_SIZE = 2_000
OUTPUT_DIR = Path("data/raw/batches")


def main() -> None:
    dataset = fetch_ucirepo(id=601)

    data = dataset.data.features.copy()
    data["Machine failure"] = dataset.data.targets["Machine failure"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_rows = len(data)

    for batch_number, start in enumerate(
        range(0, total_rows, BATCH_SIZE),
        start=1,
    ):
        end = start + BATCH_SIZE
        batch = data.iloc[start:end]

        output_file = OUTPUT_DIR / f"batch_{batch_number:03d}.csv"
        batch.to_csv(output_file, index=False)

        print(
            f"{output_file}: "
            f"{len(batch)} registros"
        )

    print("\n=== RESUMEN ===")
    print(f"Total de registros: {total_rows}")
    print(f"Tamaño del lote: {BATCH_SIZE}")
    print(f"Lotes creados: {(total_rows + BATCH_SIZE - 1) // BATCH_SIZE}")


if __name__ == "__main__":
    main()