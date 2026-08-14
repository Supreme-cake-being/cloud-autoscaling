"""
Модуль 1.1 — Генератор синтетичних даних навантаження хмарного кластера.

Навіщо потрібен:
    Реальні публічні трейси (Google Cluster Data, Alibaba Cluster Trace)
    мають розмір у десятки-сотні гігабайт і вимагають окремого завантаження.
    Для розробки та відладки пайплайну (Модулі 1-3) використовуємо
    синтетичний генератор, який відтворює ключові властивості реального
    навантаження дата-центру:
        - добова сезонність (пік вдень, спад вночі)
        - тижнева сезонність (менше навантаження у вихідні)
        - повільний тренд (зростання бази користувачів)
        - випадкові сплески (flash-crowd, вихід новини, реклама)
        - шум вимірювання

    Це дозволяє контрольовано тестувати LSTM-модель на відомих патернах
    ДО того, як переходити на реальні дані Alibaba Cluster Trace
    (інструкція з підключення реального trace — у README.md).

Формат вихідних даних відповідає полям, які реально збирає Prometheus:
    timestamp, cpu_util (%), mem_util (%), rps (requests/sec), net_in (MB/s)
"""

import numpy as np
import pandas as pd


def generate_cluster_load(
    days: int = 30,
    freq_minutes: int = 1,
    base_cpu: float = 25.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Генерує часовий ряд навантаження хмарного сервісу.

    Args:
        days: скільки днів симулювати
        freq_minutes: крок дискретизації у хвилинах (Prometheus scrape interval)
        base_cpu: базовий рівень утилізації CPU у %
        seed: seed для відтворюваності результатів (важливо для захисту роботи)

    Returns:
        DataFrame з колонками [timestamp, cpu_util, mem_util, rps, net_in]
    """
    rng = np.random.default_rng(seed)

    periods = int(days * 24 * 60 / freq_minutes)
    timestamps = pd.date_range("2025-01-01", periods=periods, freq=f"{freq_minutes}min")

    t = np.arange(periods)
    minutes_per_day = 24 * 60 / freq_minutes

    # 1. Добова сезонність: пік о 14:00-16:00, мінімум о 3:00-5:00
    daily_cycle = np.sin(2 * np.pi * (t / minutes_per_day - 0.3)) * 20

    # 2. Тижнева сезонність: менше навантаження у Сб/Нд (~-15%)
    day_of_week = (timestamps.dayofweek.to_numpy() >= 5).astype(float)
    weekly_effect = -15 * day_of_week

    # 3. Повільний лінійний тренд (зростання навантаження за місяць)
    trend = np.linspace(0, 10, periods)

    # 4. Випадкові сплески (flash-crowd) — імітують непередбачені події,
    #    саме те, що реактивний поріг НЕ встигає обробити вчасно
    spikes = np.zeros(periods)
    n_spikes = int(days * 0.4)  # ~кожні 2-3 дні
    spike_starts = rng.choice(periods, size=n_spikes, replace=False)
    for start in spike_starts:
        duration = rng.integers(10, 45)  # хвилин
        magnitude = rng.uniform(30, 70)
        end = min(start + duration, periods)
        # плавне наростання і спад сплеску (форма трикутника)
        ramp = np.linspace(0, magnitude, end - start)
        ramp = magnitude - np.abs(np.linspace(-magnitude, magnitude, end - start))
        spikes[start:end] += np.clip(ramp, 0, None)

    # 5. Шум вимірювання
    noise = rng.normal(0, 3, periods)

    cpu_util = base_cpu + daily_cycle + weekly_effect + trend + spikes + noise
    cpu_util = np.clip(cpu_util, 1, 100)

    # RPS, RAM, мережевий трафік корелюють з CPU, але не ідентичні йому
    rps = np.clip(cpu_util * rng.uniform(8, 12) + rng.normal(0, 15, periods), 0, None)
    mem_util = np.clip(0.6 * cpu_util + 20 + rng.normal(0, 4, periods), 1, 100)
    net_in = np.clip(rps * rng.uniform(0.05, 0.08) + rng.normal(0, 1, periods), 0, None)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "cpu_util": cpu_util.round(2),
            "mem_util": mem_util.round(2),
            "rps": rps.round(1),
            "net_in": net_in.round(2),
        }
    )
    return df


if __name__ == "__main__":
    df = generate_cluster_load(days=30)
    out_path = "data/synthetic_cluster_load.csv"
    df.to_csv(out_path, index=False)
    print(f"Згенеровано {len(df)} записів -> {out_path}")
    print(df.describe())