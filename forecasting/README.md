# forecasting — Модуль 2: ML-прогнозування навантаження

## Структура

```
forecasting/
├── lstm_model.py       # побудова та навчання LSTM (TensorFlow/Keras)
├── prophet_model.py     # Prophet-модель для порівняння
├── evaluate.py           # MAE/RMSE/MAPE: LSTM vs Prophet vs naive baseline
├── models/
│   ├── lstm_best.keras    # найкраща модель за val_loss (ModelCheckpoint)
│   └── prophet_model.json # серіалізована Prophet-модель (кеш)
├── results/
│   └── metrics_comparison.csv
└── README.md
```

## Запуск (з кореня проекту або з forecasting/)

```bash
pip install tensorflow prophet

cd forecasting
python lstm_model.py --epochs 50   # ~30-40 хв на CPU (з early stopping)
python evaluate.py                  # LSTM vs Prophet vs naive
```

### Швидка перевірка коду без повного навчання

```bash
python lstm_model.py --quick             # смоук-тест: 2000 прикладів, 3 епохи, ~15 сек
python lstm_model.py --epochs 10          # обмежена кількість епох
python evaluate.py --skip-prophet         # пропустити Prophet (якщо ще не встановлено)
python evaluate.py --retrain-prophet      # примусово перенавчити Prophet (ігнорує кеш)
```

## Архітектура LSTM

```
Input(window, 4) -> LSTM(u1, return_sequences=True) -> Dropout(0.2)
                  -> LSTM(u2) -> Dropout(0.2)
                  -> Dense(u2/2, relu) -> Dense(1)
```

- `window_size=60` — модель бачить останню годину навантаження
- `horizon=15` — прогнозує на 15 хв вперед
- **Auto-scaling архітектури** під обсяг train-вибірки: `LSTM(32,16)`,
  якщо прикладів менше 15000 (реальний Alibaba trace, ~8000), інакше
  `LSTM(64,32)` (синтетика, ~30000). Менша модель краще узагальнює на
  малих датасетах — класичний bias-variance компроміс.
- EarlyStopping (`--patience`, default 8) + ReduceLROnPlateau —
  захист від перенавчання, `restore_best_weights=True` відкочує до
  найкращої епохи, а не залишає останню.

## Кешування моделей

Обидві моделі зберігаються після навчання й не перенавчаються при
кожному запуску `evaluate.py`:

- `models/lstm_best.keras` — автоматично через `ModelCheckpoint`
- `models/prophet_model.json` — через `save_prophet_model()` (JSON, а
  не pickle — стабільніше між версіями CmdStan/Prophet)

## Встановлення Prophet на Windows

Prophet використовує CmdStan як бекенд. Найнадійніший шлях — версія
**1.4.0** (не 1.1.5): у старіших версіях вбудована в pip-пакет копія
CmdStan часто неповна на Windows (`AttributeError: 'Prophet' object
has no attribute 'stan_backend'`), і перевстановлення тієї самої
версії проблему не вирішує (пошкоджений артефакт "вшитий" у wheel).

```powershell
python -m pip install --upgrade cmdstanpy
python -m cmdstanpy.install_cxx_toolchain
python -m cmdstanpy.install_cmdstan --compiler

python -m pip uninstall prophet -y
python -m pip install prophet==1.4.0 --no-cache-dir
```

`evaluate.py` не падає, якщо Prophet недоступний — виводить
попередження і показує LSTM vs naive без нього (`--skip-prophet`).

## Результати (синтетичні дані, 30 днів)

| Модель            | MAE  | RMSE | MAPE, % |
| ----------------- | ---- | ---- | ------- |
| LSTM              | 3.71 | 5.00 | 15.09   |
| Naive persistence | 3.69 | 5.25 | 16.35   |
| Prophet           | 4.71 | 5.86 | 49.34   |

LSTM перевершує naive за RMSE/MAPE (краще вловлює сплески — саме
ціль методики), при паритеті по MAE. Високий MAPE у Prophet —
методологічна асиметрія: Prophet прогнозує весь тестовий період одразу
від точки навчання (multi-step-ahead), тоді як LSTM отримує свіжу
історію на кожному кроці (rolling forecast) — детальніше в docstring
`evaluate.py::evaluate_prophet()`.

## Наступний крок — Модуль 3 (scaling_logic)

Маючи навчену `models/lstm_best.keras`, переходимо до логіки
перетворення прогнозу на кількість реплік Kubernetes + реактивний
коректор на теорії черг.
