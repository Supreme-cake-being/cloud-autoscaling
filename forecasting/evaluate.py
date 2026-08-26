"""
Модуль 2.3 — Оцінка якості прогнозування: MAE, RMSE, MAPE.

Архітектурне рішення: порівнюємо ТРИ моделі, а не дві:

1. LSTM — основна модель роботи.
2. Prophet — "розумний" baseline (враховує сезонність).
3. Naive persistence — "наївний" baseline: прогноз = останнє відоме
   значення (тобто "через horizon хвилин CPU буде таким же, як зараз").
   Це МІНІМАЛЬНА планка якості: якщо LSTM не перевершує навіть цей
   тривіальний прогноз, використовувати її немає сенсу. Обов'язковий
   баseline у будь-якій roboті з часовими рядами — без нього неможливо
   довести, що складна модель дає РЕАЛЬНУ користь.

Метрики:
    MAE  (Mean Absolute Error)      — середня похибка в % CPU, легко
                                       інтерпретувати ("в середньому
                                       помиляємось на X percentage points")
    RMSE (Root Mean Squared Error)  — сильніше штрафує великі похибки;
                                       важливо для нас, бо пропущений
                                       великий сплеск коштує дорожче
                                       (порушення SLA), ніж дрібна помилка
    MAPE (Mean Absolute % Error)    — відносна похибка; обережно з нею
                                       при значеннях CPU близьких до 0
                                       (ділення на малі числа роздуває
                                       MAPE) — тому подаємо як
                                       допоміжну, а не головну метрику
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    """Рахує MAE/RMSE/MAPE, ігноруючи точки з y_true≈0 для коректного MAPE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    nonzero = np.abs(y_true) > 1e-3
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100

    return {"model": name, "MAE": round(mae, 3), "RMSE": round(rmse, 3), "MAPE_%": round(mape, 2)}


def naive_persistence_baseline(X_test: np.ndarray, target_col_idx: int = 0) -> np.ndarray:
    """
    Наївний прогноз: остання відома точка у вікні = прогноз.
    X_test форми (n_samples, window_size, n_features) -> беремо останній
    крок window'а по target-колонці.
    """
    return X_test[:, -1, target_col_idx]


def compare_models(
    y_test: np.ndarray,
    lstm_pred: np.ndarray,
    naive_pred: np.ndarray,
    scaler=None,
    target_col_idx: int = 0,
) -> pd.DataFrame:
    """
    Порівнює LSTM та naive baseline на ОДНАКОВІЙ шкалі.

    Якщо переданий scaler — денормалізуємо назад у % CPU (0-100),
    інакше метрики залишаються в нормалізованому [0,1] діапазоні
    (менш інтерпретовно, але теж коректно для відносного порівняння).
    """
    def inverse(arr):
        if scaler is None:
            return arr
        # scaler навчений на n_features колонках; для inverse_transform
        # потрібна матриця тієї ж форми — заповнюємо нулями решту колонок
        dummy = np.zeros((len(arr), scaler.n_features_in_))
        dummy[:, target_col_idx] = arr
        return scaler.inverse_transform(dummy)[:, target_col_idx]

    y_true = inverse(y_test)
    results = [
        compute_metrics(y_true, inverse(lstm_pred), "LSTM"),
        compute_metrics(y_true, inverse(naive_pred), "Naive persistence"),
    ]
    return pd.DataFrame(results)


def evaluate_prophet(test_df: pd.DataFrame, forecast: pd.DataFrame) -> dict:
    """
    Оцінює Prophet на тому самому тестовому періоді.

    ВАЖЛИВА МЕТОДОЛОГІЧНА ВІДМІННІСТЬ (обов'язково згадати в тексті
    роботи): LSTM тут оцінюється як rolling forecast — на кожному кроці
    моделі дається справжня історія останніх 60 хв, і вона прогнозує
    +15 хв вперед. Prophet у поточній реалізації натомість прогнозує
    ВЕСЬ тестовий період одразу від точки навчання (multi-step-ahead
    без "підживлення" фактичними даними по дорозі) — це складніша задача,
    тому пряме порівняння MAE/RMSE має легкий "гандикап" не на користь
    Prophet. Чесний варіант для фінальної роботи — rolling-forecast і
    для Prophet теж (перенавчання моделі на кожному кроці або періодично,
    напр. раз на день); це суттєво дорожче обчислювально, тому тут
    залишено як напрямок для подальшого уточнення експерименту.
    """
    merged = test_df.merge(forecast, on="ds", how="inner")
    return compute_metrics(merged["y"].to_numpy(), merged["yhat"].to_numpy(), "Prophet")


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, "../data_pipeline")

    parser = argparse.ArgumentParser(description="Порівняння LSTM vs Prophet vs naive baseline")
    parser.add_argument(
        "--skip-prophet", action="store_true",
        help="Пропустити Prophet (напр. якщо CmdStan не встановлено на Windows) "
             "і показати лише LSTM vs naive persistence",
    )
    parser.add_argument(
        "--retrain-prophet", action="store_true",
        help="Примусово перенавчити Prophet, навіть якщо є збережена модель "
             "у models/prophet_model.json (за замовчуванням використовується "
             "кеш, якщо він є — навчання займає ~10-15 сек, але для "
             "відтворюваності результатів у роботі краще не тренувати щоразу)",
    )
    args = parser.parse_args()

    import tensorflow as tf
    from preprocessing import clean_data, split_and_scale, create_sequences

    df = pd.read_csv("../data_pipeline/data/synthetic_cluster_load.csv", parse_dates=["timestamp"])
    cleaned = clean_data(df)
    features = ["cpu_util", "mem_util", "rps", "net_in"]
    scaled = split_and_scale(cleaned, features)

    X_test, y_test = create_sequences(scaled.test, window_size=60, horizon=15, target_col_idx=0)

    model = tf.keras.models.load_model("models/lstm_best.keras")
    lstm_pred = model.predict(X_test, verbose=0).flatten()
    naive_pred = naive_persistence_baseline(X_test)

    results = compare_models(y_test, lstm_pred, naive_pred, scaler=scaled.scaler)

    if args.skip_prophet:
        print("Prophet пропущено (--skip-prophet)")
    else:
        try:
            from prophet_model import (
                prepare_prophet_data, train_prophet, forecast_prophet,
                save_prophet_model, load_prophet_model,
            )

            train_end = int(len(cleaned) * 0.7)
            prophet_train = prepare_prophet_data(cleaned.iloc[:train_end])
            prophet_test = prepare_prophet_data(cleaned.iloc[train_end:])

            model_path = "models/prophet_model.json"
            if os.path.exists(model_path) and not args.retrain_prophet:
                print(f"Завантаження збереженої Prophet-моделі ({model_path})...")
                prophet_model = load_prophet_model(model_path)
            else:
                print("Навчання Prophet для порівняння (може зайняти хвилину)...")
                prophet_model = train_prophet(prophet_train)
                save_prophet_model(prophet_model, model_path)
                print(f"Модель збережено -> {model_path}")

            prophet_forecast = forecast_prophet(prophet_model, periods=len(prophet_test))

            prophet_metrics = evaluate_prophet(prophet_test, prophet_forecast)
            results = pd.concat([results, pd.DataFrame([prophet_metrics])], ignore_index=True)
        except Exception as e:
            # Найчастіша причина на Windows: CmdStan/C++ toolchain не
            # встановлений (див. README.md -> "Встановлення Prophet на
            # Windows"). Не валимо весь скрипт через це — показуємо
            # LSTM vs naive, а Prophet додасте пізніше повторним запуском.
            print(f"\n[!] Prophet не вдалося навчити: {type(e).__name__}: {e}")
            print("[!] Продовжуємо без Prophet. Інструкція виправлення — у README.md.")
            print("[!] Або запустіть з прапорцем --skip-prophet, щоб не бачити цю спробу.\n")

    print("\n" + results.to_string(index=False))

    os.makedirs("results", exist_ok=True)
    results.to_csv("results/metrics_comparison.csv", index=False)
    print("\nЗбережено -> results/metrics_comparison.csv")