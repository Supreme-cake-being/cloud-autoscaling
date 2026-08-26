"""
Модуль 2.2 — Prophet-модель прогнозування (baseline для порівняння з LSTM).

Навіщо Prophet у роботі, якщо є LSTM:
    Prophet (Meta/Facebook) — декомпозиційна модель (тренд + сезонність +
    свята/аномалії), розроблена спеціально для бізнес-часових рядів з
    вираженою сезонністю. Порівняння з нею дає відповідь на ключове
    питання комісії: "чи виправдана складність LSTM (нейромережа,
    довше навчання, менш інтерпретовна) порівняно з простішою
    статистичною моделлю?" Якщо LSTM не перевершує Prophet суттєво —
    це теж валідний науковий результат, який варто чесно показати.

Ключова відмінність підходу від LSTM:
    - LSTM працює з нормалізованим sliding window (numpy-масиви).
    - Prophet вимагає ІНШИЙ формат вхідних даних: DataFrame з колонками
      рівно `ds` (дата/час) та `y` (значення) — це вимога бібліотеки.
    - Prophet НЕ використовує sliding window — він моделює весь ряд
      цілісно (тренд + компоненти Фур'є для сезонності) і сам генерує
      прогноз на horizon кроків вперед через .make_future_dataframe().
    - Тому для Prophet ми беремо дані ДО sliding window (сирий часовий
      ряд з preprocessing.py, а не X_train/y_train з sequences.npz).
"""

import os

import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json


def prepare_prophet_data(df: pd.DataFrame, value_col: str = "cpu_util") -> pd.DataFrame:
    """Приводить наш формат [timestamp, cpu_util, ...] до формату Prophet [ds, y]."""
    return df[["timestamp", value_col]].rename(columns={"timestamp": "ds", value_col: "y"})


def train_prophet(train_df: pd.DataFrame) -> Prophet:
    """
    Навчає Prophet з явним вказанням сезонностей.

    daily_seasonality / weekly_seasonality=True — вмикаємо явно, а не
    покладаємось на auto-detect: наш EDA (Модуль 1) вже підтвердив
    наявність обох патернів, тому свідомо кажемо моделі їх враховувати.

    changepoint_prior_scale — контролює гнучкість тренду; менше значення
    (за замовчуванням 0.05) означає менш "смикливий" тренд, що доречно
    для короткого 30-денного ряду без довгострокових структурних зсувів.
    """
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,  # ряд лише 30 днів — річної сезонності
                                     # фізично неможливо визначити
        changepoint_prior_scale=0.05,
    )
    model.fit(train_df)
    return model


def forecast_prophet(model: Prophet, periods: int, freq: str = "1min") -> pd.DataFrame:
    """Генерує прогноз на `periods` кроків вперед з частотою `freq`."""
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def save_prophet_model(model: Prophet, path: str = "models/prophet_model.json") -> None:
    """
    Зберігає модель у JSON (НЕ pickle — Prophet явно рекомендує JSON,
    бо pickle серіалізує внутрішні Stan-об'єкти, які ламаються між
    версіями бібліотеки/CmdStan; JSON зберігає лише параметри моделі
    і стабільний навіть після оновлення Prophet).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(model_to_json(model))


def load_prophet_model(path: str = "models/prophet_model.json") -> Prophet:
    """Завантажує раніше збережену модель без повторного навчання."""
    with open(path, "r") as f:
        return model_from_json(f.read())


if __name__ == "__main__":
    df = pd.read_csv("../data_pipeline/data/synthetic_cluster_load.csv", parse_dates=["timestamp"])

    # Розбиваємо так само, як у preprocessing.py (70% train), щоб
    # результати Prophet і LSTM були порівнянні на ІДЕНТИЧНИХ періодах
    train_end = int(len(df) * 0.7)
    train_df = prepare_prophet_data(df.iloc[:train_end])
    test_df = prepare_prophet_data(df.iloc[train_end:])

    print(f"Навчання Prophet на {len(train_df)} точках...")
    model = train_prophet(train_df)

    save_prophet_model(model)
    print("Модель збережено -> models/prophet_model.json")

    forecast = forecast_prophet(model, periods=len(test_df))
    print(f"Прогноз згенеровано: {len(forecast)} точок")
    print(forecast.tail())