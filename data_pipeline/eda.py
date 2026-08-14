"""
Модуль 1.3 — Розвідувальний аналіз даних (EDA).

Мета: візуально та статистично підтвердити наявність патернів
(добова/тижнева сезонність, автокореляція), які виправдовують вибір
саме LSTM (моделі, що вміє захоплювати довгострокові залежності
в часових рядах) замість простих baseline-методів.

Ці графіки варто винести в розділ "Аналіз предметної області" роботи —
вони наочно демонструють комісії, ЧОМУ реактивний поріг не встигає:
показуємо крутизну наростання сплеску навантаження (за скільки хвилин
CPU злітає з 30% до 80%).
"""

import matplotlib
matplotlib.use("Agg")  # без GUI, збереження одразу у файл
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf

from data_generator import generate_cluster_load
from preprocessing import clean_data


def run_eda(df: pd.DataFrame, out_dir: str = "plots"):
    df = df.set_index("timestamp")

    # 1. Загальний огляд часового ряду за весь період
    fig, ax = plt.subplots(figsize=(14, 4))
    df["cpu_util"].plot(ax=ax, linewidth=0.6)
    ax.set_title("Утилізація CPU за весь період спостереження")
    ax.set_ylabel("CPU, %")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/01_full_timeseries.png", dpi=120)
    plt.close(fig)

    # 2. Добова сезонність: усереднений профіль по годинах доби
    hourly_profile = df.groupby(df.index.hour)["cpu_util"].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hourly_profile.index, hourly_profile["mean"], marker="o")
    ax.fill_between(
        hourly_profile.index,
        hourly_profile["mean"] - hourly_profile["std"],
        hourly_profile["mean"] + hourly_profile["std"],
        alpha=0.2,
    )
    ax.set_title("Середній профіль навантаження CPU по годинах доби")
    ax.set_xlabel("Година доби")
    ax.set_ylabel("CPU, %")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/02_hourly_profile.png", dpi=120)
    plt.close(fig)

    # 3. Тижнева сезонність
    dow_profile = df.groupby(df.index.dayofweek)["cpu_util"].mean()
    labels = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, dow_profile.to_numpy())
    ax.set_title("Середнє навантаження CPU по днях тижня")
    ax.set_ylabel("CPU, %")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/03_weekly_profile.png", dpi=120)
    plt.close(fig)

    # 4. Автокореляційна функція — показує, наскільки поточне значення
    #    залежить від значень N кроків тому (обґрунтування window_size)
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(df["cpu_util"], lags=180, ax=ax)
    ax.set_title("Автокореляційна функція CPU (лаги до 180 хв)")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/04_autocorrelation.png", dpi=120)
    plt.close(fig)

    # 5. Аналіз швидкості наростання сплесків — ключовий аргумент
    #    для проактивного підходу: показуємо максимальний приріст CPU
    #    за ковзне вікно 15 хв
    diff_15 = df["cpu_util"].diff(15)
    fig, ax = plt.subplots(figsize=(10, 4))
    diff_15.hist(bins=80, ax=ax)
    ax.axvline(diff_15.quantile(0.99), color="red", linestyle="--",
               label=f"99-й перцентиль: {diff_15.quantile(0.99):.1f} п.п.")
    ax.set_title("Розподіл приросту CPU за 15-хвилинне вікно")
    ax.set_xlabel("Δ CPU, percentage points")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/05_spike_growth_rate.png", dpi=120)
    plt.close(fig)

    stats = {
        "mean_cpu": df["cpu_util"].mean(),
        "std_cpu": df["cpu_util"].std(),
        "max_15min_growth": diff_15.max(),
        "p99_15min_growth": diff_15.quantile(0.99),
        "missing_values": df.isna().sum().sum(),
    }
    return stats


if __name__ == "__main__":
    raw = generate_cluster_load(days=30)
    cleaned = clean_data(raw)
    stats = run_eda(cleaned)

    print("=== Статистика EDA ===")
    for k, v in stats.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")
    print("\nГрафіки збережено у plots/")