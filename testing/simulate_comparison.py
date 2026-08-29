"""
Модуль 4.3 — Симуляційне порівняння: гібридна методика vs baseline HPA.

Рахує 5 критеріїв ефективності з п. 2.1 кваліфікаційної роботи, які
неможливо отримати зі статичного датасету (Модуль 2):
    - середнє використання процесорних ресурсів (CPU utilization)
    - середній час відгуку сервісів (latency, через M/M/c чергу)
    - кількість подій масштабування
    - частка часу роботи без перевантаження (SLA compliance)
    - загальні витрати на інфраструктуру (cost, replica-хвилини)

Методологія:
    Обидва скейлери (гібридний з Модуля 3 і baseline з testing/
    baseline_scaler.py) проганяються через ІДЕНТИЧНИЙ часовий ряд
    навантаження (той самий synthetic_cluster_load.csv, що й у
    Модулях 1-2 — забезпечує порівнянність з результатами прогнозування).

    Прогноз для проактивного рівня симулюється як РЕАЛЬНЕ майбутнє
    значення (дивимось на horizon кроків вперед у вже наявному ряді)
    + шум з амплітудою, що відповідає РЕАЛЬНІЙ похибці навченої LSTM
    (MAE з Модуля 2, evaluate.py) — це чесніше, ніж вигадувати довільну
    точність прогнозу: використовуємо вже виміряну якість власної моделі.

    Обидва скейлери підпорядковані ОДНАКОВІЙ затримці старту поду
    (PodFleetSimulator) — це єдина справедлива умова порівняння:
    різниця в результатах походить виключно від того, ЧИ скейлер знав
    про навантаження заздалегідь, а не від штучної переваги одного
    підходу над іншим.
"""

import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, "../scaling_logic")
from config import ScalingConfig
from decision_engine import ProactiveDecisionEngine
from queuing_corrector import ReactiveCorrector, expected_wait_time

from baseline_scaler import BaselineThresholdScaler, BaselineConfig
from pod_fleet import PodFleetSimulator

# --- Параметри симуляції ---
DEFAULT_STARTUP_DELAY_MINUTES = 3   # типове значення для основного порівняння —
                                     # середина заявленого в ТЗ діапазону "1-5 хв",
                                     # не найвигідніший і не найгірший край
FORECAST_HORIZON_MINUTES = 15       # той самий horizon, що й у Модулі 2
PROACTIVE_CYCLE_MINUTES = 5         # як часто ОНОВЛЮЄТЬСЯ прогноз (реалістична
                                     # каденція — LSTM не перерахує прогноз
                                     # щохвилини; реактивний коректор далі
                                     # працює кожен крок на реальних метриках,
                                     # саме так задокументовано в autoscaler.py)
LSTM_MAE_RPS = 37.0                 # похибка прогнозу в RPS-еквіваленті:
                                     # MAE з Модуля 2 (~3.7 percentage points
                                     # CPU) * 10 (той самий коефіцієнт
                                     # cpu->rps, що й у синтетичному генераторі)
COST_PER_REPLICA_MINUTE = 0.05 / 60  # умовна вартість (напр. $0.05/год за под)


def simulate_hybrid(df: pd.DataFrame, scaling_config: ScalingConfig, seed: int = 42,
                     startup_delay_minutes: int = DEFAULT_STARTUP_DELAY_MINUTES) -> dict:
    """Гібридна методика: проактивний прогноз (з шумом=реальний MAE) + реактивний коректор."""
    rng = np.random.default_rng(seed)

    proactive = ProactiveDecisionEngine(scaling_config)
    reactive = ReactiveCorrector(scaling_config)
    fleet = PodFleetSimulator(startup_delay_steps=startup_delay_minutes, active_replicas=scaling_config.min_replicas)

    rps = df["rps"].to_numpy()
    n = len(rps)
    t0 = datetime(2025, 1, 1)

    active_history, commanded_history, wait_history = [], [], []
    current_proactive_replicas = scaling_config.min_replicas

    for i in range(n):
        current_time = t0 + timedelta(minutes=i)

        # Проактивний рівень оновлює рішення лише раз на PROACTIVE_CYCLE_MINUTES —
        # реалістична каденість перерахунку прогнозу, а не щохвилинне
        # "тремтіння" від незалежного шуму на кожному кроці
        if i % PROACTIVE_CYCLE_MINUTES == 0:
            future_idx = min(i + FORECAST_HORIZON_MINUTES, n - 1)
            predicted = max(0.0, rps[future_idx] + rng.normal(0, LSTM_MAE_RPS))
            proactive_decision = proactive.decide(predicted, current_time)
            current_proactive_replicas = proactive_decision.replicas

        # Реактивний коректор працює КОЖЕН крок на РЕАЛЬНИХ поточних метриках
        correction = reactive.correct(rps[i], current_proactive_replicas)

        commanded_history.append(correction.replicas)
        active = fleet.step(correction.replicas, i)
        active_history.append(active)

        wait = expected_wait_time(max(active, 1), rps[i], scaling_config.service_rate_per_replica)
        wait_history.append(wait)

    return _compute_metrics(active_history, commanded_history, wait_history, rps, scaling_config)


def simulate_baseline(df: pd.DataFrame, baseline_config: BaselineConfig, scaling_config: ScalingConfig,
                       startup_delay_minutes: int = DEFAULT_STARTUP_DELAY_MINUTES) -> dict:
    """Baseline: чисто реактивний threshold-скейлер, без прогнозу."""
    scaler = BaselineThresholdScaler(baseline_config)
    fleet = PodFleetSimulator(startup_delay_steps=startup_delay_minutes, active_replicas=baseline_config.min_replicas)

    rps = df["rps"].to_numpy()
    n = len(rps)
    t0 = datetime(2025, 1, 1)

    active_history, commanded_history, wait_history = [], [], []

    for i in range(n):
        current_time = t0 + timedelta(minutes=i)
        commanded = scaler.decide(rps[i], current_time)  # бачить лише ПОТОЧНЕ навантаження

        commanded_history.append(commanded)
        active = fleet.step(commanded, i)
        active_history.append(active)

        wait = expected_wait_time(max(active, 1), rps[i], scaling_config.service_rate_per_replica)
        wait_history.append(wait)

    return _compute_metrics(active_history, commanded_history, wait_history, rps, scaling_config)


def _compute_metrics(active_history, commanded_history, wait_history, rps, scaling_config) -> dict:
    active = np.array(active_history)
    wait = np.array(wait_history)

    utilization = np.clip(rps / (np.maximum(active, 1) * scaling_config.capacity_per_replica_rps), 0, 2)
    scaling_events = sum(1 for i in range(1, len(commanded_history)) if commanded_history[i] != commanded_history[i - 1])
    sla_ok = wait <= scaling_config.max_wait_time_sec
    finite_wait = wait[np.isfinite(wait)]

    return {
        "avg_cpu_utilization_%": round(float(np.mean(utilization)) * 100, 1),
        "avg_latency_sec": round(float(np.mean(finite_wait)) if len(finite_wait) else float("inf"), 4),
        "p95_latency_sec": round(float(np.percentile(finite_wait, 95)) if len(finite_wait) else float("inf"), 4),
        "scaling_events": scaling_events,
        "sla_compliance_%": round(float(np.mean(sla_ok)) * 100, 2),
        "total_cost_usd": round(float(np.sum(active)) * COST_PER_REPLICA_MINUTE, 2),
        "avg_replicas": round(float(np.mean(active)), 2),
    }


def run_sensitivity_analysis(df: pd.DataFrame, delays: list[int] = (1, 2, 3, 4, 5)) -> pd.DataFrame:
    """
    Прогонить обидва методи по всьому заявленому в ТЗ діапазону затримки
    старту поду (1-5 хв), а не по одній довільно обраній точці.

    Навіщо: перевага проактивного підходу МЕХАНІЧНО залежить від того,
    наскільки довго реактивний скейлер "наздоганяє" сплеск наосліп —
    чим коротша затримка старту, тим менше реактивний підхід втрачає
    від відсутності передбачення. Показ ОДНОГО числа (напр. лише 2 хв)
    методологічно некоректний — легко випадково обрати точку, вигідну
    чи невигідну одному з методів. Повний діапазон чесно показує, ЗА
    ЯКИХ УМОВ методика дає перевагу, а за яких — ні.
    """
    scaling_config = ScalingConfig()
    scaling_config.safety_factor = 1.0 / 0.7
    baseline_config = BaselineConfig(
        capacity_per_replica_rps=scaling_config.capacity_per_replica_rps,
        min_replicas=scaling_config.min_replicas,
        max_replicas=scaling_config.max_replicas,
    )

    rows = []
    for delay in delays:
        h = simulate_hybrid(df, scaling_config, startup_delay_minutes=delay)
        b = simulate_baseline(df, baseline_config, scaling_config, startup_delay_minutes=delay)
        rows.append({"startup_delay_min": delay, "метод": "Гібридна методика", **h})
        rows.append({"startup_delay_min": delay, "метод": "Baseline (threshold HPA)", **b})

    return pd.DataFrame(rows)


def plot_sensitivity(sensitivity_df: pd.DataFrame, out_path: str = "results/sensitivity_analysis.png", tz_boundary: int = 5):
    """
    Графік: як SLA compliance і вартість змінюються залежно від startup_delay.

    tz_boundary: межа "типового" діапазону, заявленого в ТЗ (5 хв) —
    позначається вертикальною лінією, щоб було видно, де закінчується
    сценарій з тексту роботи і починається розширена перевірка стійкості
    висновків за його межами (довші cold-start, напр. підняття нового
    вузла кластера).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"Гібридна методика": "#4C72B0", "Baseline (threshold HPA)": "#DD8452"}

    for method, color in colors.items():
        sub = sensitivity_df[sensitivity_df["метод"] == method].sort_values("startup_delay_min")
        axes[0].plot(sub["startup_delay_min"], sub["sla_compliance_%"], marker="o", label=method, color=color)
        axes[1].plot(sub["startup_delay_min"], sub["total_cost_usd"], marker="o", label=method, color=color)

    max_delay = sensitivity_df["startup_delay_min"].max()
    for ax in axes:
        if max_delay > tz_boundary:
            ax.axvline(tz_boundary, color="gray", linestyle="--", linewidth=1, alpha=0.7)
            ax.text(tz_boundary + 0.1, ax.get_ylim()[1] if ax is axes[0] else ax.get_ylim()[0],
                     f"межа ТЗ ({tz_boundary} хв)", fontsize=7, color="gray", va="top" if ax is axes[0] else "bottom")

    axes[0].set_title("SLA compliance vs. затримка старту поду")
    axes[0].set_xlabel("Затримка старту поду, хв")
    axes[0].set_ylabel("SLA compliance, %")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Вартість vs. затримка старту поду")
    axes[1].set_xlabel("Затримка старту поду, хв")
    axes[1].set_ylabel("Вартість, $")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Аналіз чутливості: за яких умов проактивний підхід дає перевагу", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Порівняння гібридної методики з baseline HPA")
    parser.add_argument(
        "--max-delay", type=int, default=10,
        help="Максимальна затримка старту поду (хв) для аналізу чутливості (default: 10). "
             "ТЗ заявляє діапазон '1-5 хв' як типовий, але реальні cold-start сценарії "
             "(підняття нового вузла кластера, великий образ контейнера) можуть бути довшими — "
             "розширений діапазон перевіряє стійкість висновків за межами типового сценарію.",
    )
    args = parser.parse_args()

    df = pd.read_csv("../data_pipeline/data/synthetic_cluster_load.csv", parse_dates=["timestamp"])
    print(f"Симуляція на {len(df)} хвилинах навантаження ({len(df) / 1440:.0f} днів)\n")

    scaling_config = ScalingConfig()
    # Вирівнюємо запас (safety margin) між методами: baseline закладає
    # запас через target_utilization (1/0.7 ≈ 1.43), тому проактивний
    # safety_factor встановлюємо на те саме значення — інакше різниця
    # в результатах пояснювалась би різним рівнем "перестраховки" двох
    # методів, а не тим, що дає ЗНАННЯ майбутнього (суть порівняння).
    scaling_config.safety_factor = 1.0 / 0.7
    baseline_config = BaselineConfig(
        capacity_per_replica_rps=scaling_config.capacity_per_replica_rps,
        min_replicas=scaling_config.min_replicas,
        max_replicas=scaling_config.max_replicas,
    )

    print(f"=== Основне порівняння (startup_delay={DEFAULT_STARTUP_DELAY_MINUTES} хв — середина "
          f"заявленого в ТЗ діапазону 1-5 хв) ===\n")

    print("Симуляція гібридної методики (проактивний + реактивний)...")
    hybrid_metrics = simulate_hybrid(df, scaling_config)

    print("Симуляція baseline (порогове реактивне масштабування)...\n")
    baseline_metrics = simulate_baseline(df, baseline_config, scaling_config)

    results = pd.DataFrame([
        {"метод": "Гібридна методика", **hybrid_metrics},
        {"метод": "Baseline (threshold HPA)", **baseline_metrics},
    ])

    print(results.to_string(index=False))

    import os
    os.makedirs("results", exist_ok=True)
    results.to_csv("results/comparison_metrics.csv", index=False)
    print("\nЗбережено -> results/comparison_metrics.csv")

    # --- Графік основного порівняння ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_to_plot = [
        ("scaling_events", "Кількість подій\nмасштабування", "менше=краще"),
        ("sla_compliance_%", "SLA compliance, %", "більше=краще"),
        ("total_cost_usd", "Вартість, $", "менше=краще"),
        ("avg_cpu_utilization_%", "Утилізація CPU, %", "орієнтир: ближче\nдо 70%"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    labels = results["метод"].tolist()
    colors = ["#4C72B0", "#DD8452"]

    for ax, (col, title, note) in zip(axes, metrics_to_plot):
        values = results[col].tolist()
        ax.bar(labels, values, color=colors)
        ax.set_title(f"{title}\n({note})", fontsize=10)
        ax.tick_params(axis="x", rotation=15, labelsize=8)
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:g}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"Гібридна методика vs Baseline (startup_delay={DEFAULT_STARTUP_DELAY_MINUTES} хв)", fontsize=12)
    fig.tight_layout()
    fig.savefig("results/comparison_chart.png", dpi=120)
    print("Графік збережено -> results/comparison_chart.png")

    # --- Sensitivity-аналіз: розширений діапазон, не лише типовий 1-5 хв ---
    print(f"\n=== Аналіз чутливості: startup_delay від 1 до {args.max_delay} хв ===\n")
    sensitivity_df = run_sensitivity_analysis(df, delays=range(1, args.max_delay + 1))

    pivot = sensitivity_df.pivot(index="startup_delay_min", columns="метод", values=["sla_compliance_%", "total_cost_usd", "scaling_events"])
    print(pivot.to_string())

    sensitivity_df.to_csv("results/sensitivity_analysis.csv", index=False)
    plot_sensitivity(sensitivity_df)
    print("\nЗбережено -> results/sensitivity_analysis.csv, results/sensitivity_analysis.png")

    # Точка перетину: з якого startup_delay гібрид починає вигравати по SLA
    hybrid_sla = sensitivity_df[sensitivity_df["метод"] == "Гібридна методика"].set_index("startup_delay_min")["sla_compliance_%"]
    baseline_sla = sensitivity_df[sensitivity_df["метод"] == "Baseline (threshold HPA)"].set_index("startup_delay_min")["sla_compliance_%"]
    crossover = [d for d in hybrid_sla.index if hybrid_sla[d] > baseline_sla[d]]
    if crossover:
        print(f"\nГібридна методика перевершує baseline по SLA, починаючи з startup_delay = {min(crossover)} хв")
    else:
        print("\nУ перевіреному діапазоні (1-5 хв) гібридна методика не перевершує baseline по SLA "
              "— варто розширити діапазон або переглянути параметри моделі.")