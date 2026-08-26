"""
Модуль 2.1 — LSTM-модель прогнозування навантаження.

Архітектурне рішення:

Чому LSTM, а не проста RNN чи GRU:
    - Проста RNN страждає від vanishing gradient на довших послідовностях
      (у нас window_size=60 кроків) — LSTM вирішує це через gate-механізм
      (forget/input/output gates), який контролює, яку інформацію
      "пам'ятати" довгостроково.
    - GRU простіша (менше параметрів, швидше навчається), але LSTM дає
      трохи кращу точність на довших залежностях (добова сезонність =
      1440 хв, тижнева = 10080 хв) — у нашому вікні 60 хв ми бачимо лише
      частину добового циклу, тож здатність LSTM зберігати
      довгострокові патерни важлива.

Архітектура мережі (обґрунтування для захисту):
    Input(window_size=60, n_features=4)
      -> LSTM(64, return_sequences=True)   # перший шар "бачить" всю
                                              # послідовність, повертає
                                              # вихід для КОЖНОГО кроку
      -> Dropout(0.2)                        # регуляризація проти
                                              # перенавчання
      -> LSTM(32)                            # другий шар стискає в один
                                              # вектор ознак (контекст)
      -> Dropout(0.2)
      -> Dense(16, relu)                     # невеликий "змішувальний"
                                              # шар перед виходом
      -> Dense(1)                            # прогноз одного значення
                                              # (CPU через horizon хв)

    Два LSTM-шари (stacked LSTM) обрані замість одного глибокого —
    типовий компроміс "глибина vs здатність узагальнювати" для рядів
    середньої складності; один шар LSTM(128) недонавчається на наших
    патернах, три шари — перенавчаються на 30-денному синтетичному ряді.

Функція втрат — MSE (Mean Squared Error): штрафує великі відхилення
    сильніше за малі, що доречно для задачі — краще передбачити пік
    з невеликою похибкою, ніж повністю його пропустити.
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_model(window_size: int, n_features: int, units: tuple = (64, 32)) -> keras.Model:
    """Будує stacked LSTM-модель для прогнозування одного кроку вперед.

    Args:
        units: (units_layer1, units_layer2) — розмір LSTM-шарів.
               Більше юнітів = більша ємність моделі, але й більший
               ризик перенавчання на малих датасетах (див. train_lstm
               про auto-scaling під розмір train-вибірки).
    """
    u1, u2 = units
    model = keras.Sequential([
        layers.Input(shape=(window_size, n_features)),
        layers.LSTM(u1, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(u2),
        layers.Dropout(0.2),
        layers.Dense(max(u2 // 2, 8), activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    return model


def _auto_select_units(n_train_samples: int) -> tuple:
    """
    Автоматично обирає розмір LSTM під обсяг train-вибірки.

    Чому це важливо: реальний Alibaba trace (v2018) охоплює лише ~8
    днів (~8000 sliding-window прикладів після split), тоді як
    синтетичний генератор дає 30 днів (~30000 прикладів). Архітектура
    (64, 32), що добре працює на синтетиці, на короткому реальному
    ряді перенавчається/недонавчається — замало прикладів, щоб
    ~50К+ параметрів моделі знайшли стійкий патерн. Менша модель
    (32, 16) має менше параметрів і краще узагальнює на малих вибірках
    — класичний компроміс bias-variance, який варто явно згадати
    в тексті роботи при поясненні різниці результатів на синтетичних
    і реальних даних.
    """
    if n_train_samples < 15_000:
        return (32, 16)
    return (64, 32)


def train_lstm(
    X_train, y_train, X_val, y_val,
    window_size: int, n_features: int,
    epochs: int = 50, batch_size: int = 64,
    model_path: str = "models/lstm_best.keras",
    units: tuple | None = None,
    patience: int = 8,
):
    """
    Навчає модель з EarlyStopping (зупинка, коли val_loss перестає
    покращуватись — захист від перенавчання) та ModelCheckpoint
    (зберігає найкращу за val_loss версію, а не останню епоху).

    units: якщо None — розмір LSTM обирається автоматично залежно від
           обсягу train-вибірки (див. _auto_select_units). Передайте
           явно (напр. (64, 32)), щоб зафіксувати архітектуру незалежно
           від розміру даних.
    """
    if units is None:
        units = _auto_select_units(len(X_train))
        print(f"Auto-обрана архітектура під {len(X_train)} train-прикладів: LSTM{units}")

    model = build_lstm_model(window_size, n_features, units=units)

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(patience // 2, 2), min_lr=1e-5),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=2,
    )
    return model, history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Навчання LSTM-моделі прогнозування навантаження")
    parser.add_argument("--epochs", type=int, default=50, help="Максимальна кількість епох (default: 50)")
    parser.add_argument("--batch-size", type=int, default=64, help="Розмір батчу (default: 64)")
    parser.add_argument(
        "--units", type=int, nargs=2, default=None, metavar=("U1", "U2"),
        help="Розмір LSTM-шарів, напр. --units 32 16. Якщо не задано — "
             "обирається автоматично залежно від обсягу train-вибірки.",
    )
    parser.add_argument(
        "--patience", type=int, default=8,
        help="EarlyStopping patience — скільки епох без покращення val_loss "
             "чекати перед зупинкою (default: 8). Збільшіть (напр. до 15-20) "
             "для малих датасетів (реальний Alibaba trace), де val_loss "
             "коливається сильніше через менший обсяг даних.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Швидкий смоук-тест: 2000 train / 500 val прикладів, 3 епохи — "
             "перевірити, що пайплайн працює, за секунди замість 30-40 хв",
    )
    args = parser.parse_args()

    data = np.load("sequences.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    epochs = args.epochs
    if args.quick:
        X_train, y_train = X_train[:2000], y_train[:2000]
        X_val, y_val = X_val[:500], y_val[:500]
        epochs = min(epochs, 3)
        print("=== ШВИДКИЙ РЕЖИМ (--quick): урізана вибірка, лише для перевірки пайплайну ===")

    window_size, n_features = X_train.shape[1], X_train.shape[2]
    print(f"Навчання LSTM: window_size={window_size}, n_features={n_features}, epochs={epochs}")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    model, history = train_lstm(
        X_train, y_train, X_val, y_val, window_size, n_features,
        epochs=epochs, batch_size=args.batch_size,
        units=tuple(args.units) if args.units else None,
        patience=args.patience,
    )

    print(f"\nНайкращий val_loss: {min(history.history['val_loss']):.5f}")
    print(f"Навчено епох (до early stopping): {len(history.history['loss'])}")