"""
Дослідження впливу розміру LSTM на якість прогнозування (на кількох seed).

Питання: чи справедливе рішення з підрозділу 3.3 застосовувати меншу
архітектуру LSTM(32, 16) на малому реальному трейсі (~8000 навчальних
прикладів) замість LSTM(64, 32)?

Чому кілька запусків, а не один: навчання нейромережі стохастичне, і різниця
між двома запусками однієї конфігурації може бути співмірна з різницею між
конфігураціями. Тому кожну конфігурацію навчаємо на кількох seed і порівнюємо
середнє та розкид.

Протокол ідентичний основному експерименту: ті ж дані й препроцесинг, та сама
train_lstm (EarlyStopping, ReduceLROnPlateau), та сама оцінка на тестовій
вибірці (compare_models з evaluate.py). Скрипт пише лише власні файли
results/architecture_study_<dataset>.* і не торкається models/ та metrics_*.csv.

Використання (з теки forecasting/):
    python architecture_study.py                      # alibaba, 32,16 vs 64,32, 5 seed
    python architecture_study.py --seeds 3 --configs 16,8 32,16 64,32
"""

import argparse
import os
import sys
import time

# TensorFlow — ПЕРШИМ, до numpy/pandas/scikit-learn (порядок імпортів має
# значення на Windows: sklearn перед TensorFlow ламає завантаження DLL).
import tensorflow as tf  # noqa: F401
from tensorflow import keras

import numpy as np
import pandas as pd

sys.path.insert(0, "../data_pipeline")
from preprocessing import clean_data, split_and_scale, create_sequences  # noqa: E402
from lstm_model import train_lstm  # noqa: E402
from evaluate import compare_models, naive_persistence_baseline  # noqa: E402

DATA_PATHS = {
    "synthetic": "../data_pipeline/data/synthetic_cluster_load.csv",
    "alibaba": "../data_pipeline/data/alibaba_cluster_load.csv",
}
FEATURES = ["cpu_util", "mem_util", "rps", "net_in"]
WINDOW, HORIZON = 60, 15


def parse_config(text: str) -> tuple:
    u1, u2 = text.split(",")
    return int(u1), int(u2)


def run_study(dataset, configs, n_seeds, epochs, patience):
    df = clean_data(pd.read_csv(DATA_PATHS[dataset], parse_dates=["timestamp"]))
    scaled = split_and_scale(df, FEATURES)
    X_train, y_train = create_sequences(scaled.train, WINDOW, HORIZON)
    X_val, y_val = create_sequences(scaled.val, WINDOW, HORIZON)
    X_test, y_test = create_sequences(scaled.test, WINDOW, HORIZON)
    print(f"Набір: {dataset}; train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")

    naive_pred = naive_persistence_baseline(X_test)
    tmp_model = f"models/_study_tmp_{dataset}.keras"
    rows, naive_row = [], None

    for units in configs:
        for seed in range(n_seeds):
            keras.utils.set_random_seed(seed)
            started = time.time()
            model, history = train_lstm(
                X_train, y_train, X_val, y_val, WINDOW, X_train.shape[2],
                epochs=epochs, model_path=tmp_model, units=units, patience=patience,
            )
            elapsed = time.time() - started
            pred = model.predict(X_test, verbose=0).flatten()
            table = compare_models(y_test, pred, naive_pred, scaler=scaled.scaler)
            lstm, naive_row = table.iloc[0], table.iloc[1]
            val_curve = history.history["val_loss"]
            rows.append({
                "config": f"LSTM({units[0]},{units[1]})", "seed": seed,
                "epochs_run": len(val_curve), "best_epoch": int(np.argmin(val_curve)) + 1,
                "best_val_loss": round(float(min(val_curve)), 5),
                "MAE": lstm["MAE"], "RMSE": lstm["RMSE"], "MAPE_%": lstm["MAPE_%"],
                "train_sec": round(elapsed, 1),
            })
            print(f">>> {rows[-1]['config']} seed={seed}: MAE={lstm['MAE']} RMSE={lstm['RMSE']} "
                  f"MAPE={lstm['MAPE_%']} (епох {len(val_curve)}, {elapsed:.0f} с)")

    if os.path.exists(tmp_model):
        os.remove(tmp_model)
    return pd.DataFrame(rows), naive_row


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = ["best_val_loss", "MAE", "RMSE", "MAPE_%", "best_epoch", "train_sec"]
    mean, std = runs.groupby("config")[metrics].mean(), runs.groupby("config")[metrics].std(ddof=1)
    out = pd.DataFrame(index=mean.index)
    for m in metrics:
        out[m] = [f"{a:.4g} ± {0 if np.isnan(b) else b:.2g}" for a, b in zip(mean[m], std[m])]
    return out


def plot(runs, naive_row, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = list(dict.fromkeys(runs["config"]))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, metric in zip(axes, ["MAE", "RMSE"]):
        for i, cfg in enumerate(configs):
            vals = runs.loc[runs["config"] == cfg, metric]
            ax.scatter([i] * len(vals), vals, alpha=0.7)
            ax.hlines(vals.mean(), i - 0.25, i + 0.25, colors="black")
        ax.axhline(naive_row[metric], color="red", linestyle="--", label="Naive persistence")
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs)
        ax.set_title(f"{metric} на тестовій вибірці (точка = один seed)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Порівняння архітектур LSTM на кількох seed")
    parser.add_argument("--dataset", choices=list(DATA_PATHS), default="alibaba")
    parser.add_argument("--configs", nargs="+", default=["32,16", "64,32"],
                        help="Конфігурації 'u1,u2', напр. 32,16 64,32")
    parser.add_argument("--seeds", type=int, default=5, help="Кількість seed на конфігурацію")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    runs, naive_row = run_study(args.dataset, [parse_config(c) for c in args.configs],
                                args.seeds, args.epochs, args.patience)

    os.makedirs("results", exist_ok=True)
    base = f"results/architecture_study_{args.dataset}"
    runs.to_csv(f"{base}.csv", index=False)
    plot(runs, naive_row, f"{base}.png")

    print("\n=== Середнє ± стандартне відхилення по seed ===")
    print(summarize(runs).to_string())
    print(f"\nNaive persistence (еталон): MAE={naive_row['MAE']} RMSE={naive_row['RMSE']} "
          f"MAPE={naive_row['MAPE_%']}")
    print(f"\nЗбережено -> {base}.csv, {base}.png")