# forecasting — Модуль 2: ML-прогнозування навантаження

## Структура

```
forecasting/
├── lstm_model.py       # побудова та навчання LSTM (TensorFlow/Keras)
├── prophet_model.py     # Prophet-модель для порівняння
├── evaluate.py           # MAE/RMSE/MAPE: LSTM vs Prophet vs naive baseline
├── sequences.npz         # копія з data_pipeline/data/ (X_train/y_train тощо)
├── models/
│   └── lstm_best.keras   # найкраща модель за val_loss (ModelCheckpoint)
├── results/
│   └── metrics_comparison.csv
└── README.md
```

## Запуск (з кореня проекту або з forecasting/)

```bash
pip install tensorflow prophet

cd forecasting
python lstm_model.py     # ~30-40 хв на CPU (50 епох з early stopping)
python evaluate.py        # LSTM vs Prophet vs naive, ~1-2 хв (Prophet fit)
```

### Швидка перевірка коду без повного навчання

```bash
python lstm_model.py --quick             # смоук-тест: 2000 прикладів, 3 епохи, ~15 сек
python lstm_model.py --epochs 10          # обмежена кількість епох
python evaluate.py --skip-prophet         # пропустити Prophet (якщо ще не встановлено)
python evaluate.py --retrain-prophet      # примусово перенавчити Prophet (ігнорує кеш)
```

### Кешування моделей

Обидві моделі зберігаються після навчання й **не перенавчаються при
кожному запуску** `evaluate.py`:

- `models/lstm_best.keras` — зберігається автоматично через
  `ModelCheckpoint` під час `python lstm_model.py`
- `models/prophet_model.json` — зберігається через `save_prophet_model()`
  (JSON, а не pickle — Prophet явно рекомендує JSON, бо pickle серіалізує
  внутрішні Stan-об'єкти, які ламаються між версіями CmdStan/Prophet)

Якщо `models/prophet_model.json` вже існує, `evaluate.py` завантажує
його замість повторного навчання (~10-15 сек економиться щоразу).
Використайте `--retrain-prophet`, якщо змінили дані чи гіперпараметри
Prophet і потрібна свіжа модель.

### Встановлення Prophet на Windows

Prophet використовує CmdStan (компілятор Stan) як бекенд. Крок його
автоматичного встановлення часто мовчки не спрацьовує на Windows через
відсутність C++ компілятора — після `pip install prophet` перший запуск
падає з `AttributeError: 'Prophet' object has no attribute 'stan_backend'`.

Виправлення (10-20 хв, компілюється C++ код):

```powershell
pip install --upgrade cmdstanpy

python -m cmdstanpy.install_cxx_toolchain
python -m cmdstanpy.install_cmdstan --compiler
```

Якщо не допомагає (корпоративні обмеження/антивірус блокують
компіляцію) — надійніший варіант: виконати саме цей крок через WSL2
(Ubuntu), де компіляція зазвичай проходить без додаткових танців з
тулчейнами.

**`evaluate.py` не падає, якщо Prophet недоступний** — виводить
попередження і показує порівняння LSTM vs naive baseline без нього. Не
блокує подальшу роботу над проектом, доки виправляєте середовище.

## Архітектура LSTM

```
Input(60, 4) -> LSTM(64, return_sequences=True) -> Dropout(0.2)
             -> LSTM(32) -> Dropout(0.2)
             -> Dense(16, relu) -> Dense(1)
```

- `window_size=60` — модель бачить останню годину навантаження
- `horizon=15` — прогнозує на 15 хв вперед (достатньо, щоб встигнути
  "розігріти" новий под, який стартує 1-5 хв)
- EarlyStopping + ReduceLROnPlateau — захист від перенавчання

## Результати (попередній прогін, синтетичні дані)

| Модель            | MAE  | RMSE | MAPE, % |
| ----------------- | ---- | ---- | ------- |
| LSTM              | 3.21 | 4.44 | 13.74   |
| Naive persistence | 3.69 | 5.25 | 16.35   |
| Prophet           | 4.82 | 5.97 | 50.41   |

**Важливо для тексту роботи:** Prophet у цій реалізації оцінюється як
multi-step-ahead прогноз від точки навчання (без "підживлення"
фактичними даними по дорозі), тоді як LSTM — rolling forecast (справжня
історія на кожному кроці). Це не повністю "чесне" порівняння в один клік
— пояснення й напрямок для rolling-оцінки Prophet є в docstring
`evaluate.py::evaluate_prophet()`. Якщо є час до захисту, це перше, що
варто вдосконалити для більш строгого порівняння.

## Наступний крок — Модуль 3 (scaling_logic)

Маючи навчену `models/lstm_best.keras`, переходимо до логіки
перетворення прогнозу на кількість реплік Kubernetes + реактивний
коректор на основі теорії черг.
