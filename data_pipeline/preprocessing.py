"""
Модуль 1.2 — Очистка, нормалізація та формування sliding window.

Архітектурні рішення:

1. Очистка (clean_data):
   - Реальні метрики Prometheus можуть мати пропуски (scrape failed) та
     викиди (aномальні значення через збій моніторингу). Пропуски
     заповнюємо лінійною інтерполяцією (навантаження — це неперервний
     процес, різкий "стрибок в нуль" фізично малоймовірний).
   - Викиди детектуємо через IQR-метод і теж інтерполюємо, а НЕ видаляємо
     рядок — бо для часового ряду видалення рядка ламає рівномірний крок
     дискретизації, критичний для LSTM.

2. Нормалізація (MinMaxScaler, діапазон [0,1]):
   - LSTM з sigmoid/tanh активаціями чутлива до масштабу входів.
   - Обов'язково: scaler навчається ТІЛЬКИ на train-частині, а потім
     застосовується до val/test — інакше буде витік інформації
     (data leakage) з майбутнього в минуле, що штучно завищить якість
     моделі на захисті.

3. Sliding window (create_sequences):
   - LSTM приймає вхід форми (n_samples, window_size, n_features).
   - window_size = скільки минулих кроків модель бачить, щоб зробити
     прогноз на horizon кроків вперед.
   - Приклад: window_size=60 (останні 60 хв), horizon=15 (прогноз на
     15 хв вперед) — типові параметри для проактивного масштабування,
     бо запуск нового поду займає 1-5 хв, тож 15-хвилинного
     попередження достатньо для "розігріву" ресурсів.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def clean_data(df: pd.DataFrame, iqr_multiplier: float = 3.0) -> pd.DataFrame:
    """Заповнює пропуски та згладжує викиди через IQR-метод."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
        outliers = (df[col] < lower) | (df[col] > upper)
        df.loc[outliers, col] = np.nan  # позначаємо викиди як пропуски

    df[numeric_cols] = df[numeric_cols].interpolate(method="linear").bfill().ffill()
    return df


@dataclass
class ScaledDataset:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    scaler: MinMaxScaler
    feature_names: list


def split_and_scale(
    df: pd.DataFrame,
    feature_cols: list,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> ScaledDataset:
    """
    Розбиває часовий ряд на train/val/test БЕЗ перемішування (shuffle=False
    є критично важливим для часових рядів — інакше майбутнє "просочується"
    у навчання) і масштабує в [0,1].
    """
    data = df[feature_cols].to_numpy()
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    scaler = MinMaxScaler()
    train = scaler.fit_transform(data[:train_end])
    val = scaler.transform(data[train_end:val_end])
    test = scaler.transform(data[val_end:])

    return ScaledDataset(train=train, val=val, test=test, scaler=scaler, feature_names=feature_cols)


def create_sequences(
    data: np.ndarray,
    window_size: int = 60,
    horizon: int = 15,
    target_col_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Формує (X, y) для навчання LSTM.

    X[i] = data[i : i+window_size]                       -> форма (window_size, n_features)
    y[i] = data[i+window_size+horizon-1, target_col_idx]  -> скаляр (прогноз на horizon вперед)

    Returns:
        X: (n_samples, window_size, n_features)
        y: (n_samples,)
    """
    X, y = [], []
    n = len(data)
    for i in range(n - window_size - horizon + 1):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size + horizon - 1, target_col_idx])
    return np.array(X), np.array(y)


if __name__ == "__main__":
    # Перемикач джерела даних: підставте "data/alibaba_cluster_load.csv",
    # коли запустите load_alibaba_data.py — решта пайплайну не зміниться,
    # бо обидва файли дають ідентичний формат CSV.
    DATA_SOURCE = "data/alibaba_cluster_load.csv"  # synthetic_cluster_load.csv

    raw = pd.read_csv(DATA_SOURCE, parse_dates=["timestamp"])
    cleaned = clean_data(raw)

    features = ["cpu_util", "mem_util", "rps", "net_in"]
    scaled = split_and_scale(cleaned, features)

    X_train, y_train = create_sequences(scaled.train, window_size=60, horizon=15, target_col_idx=0)
    X_val, y_val = create_sequences(scaled.val, window_size=60, horizon=15, target_col_idx=0)
    X_test, y_test = create_sequences(scaled.test, window_size=60, horizon=15, target_col_idx=0)

    print(f"Train: X{X_train.shape} y{y_train.shape}")
    print(f"Val:   X{X_val.shape} y{y_val.shape}")
    print(f"Test:  X{X_test.shape} y{y_test.shape}")

    np.savez(
        "data/sequences.npz",
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )
    print("Збережено -> data/sequences.npz")