# forecasting — Модуль 2: ML-прогнозування навантаження

## Структура

```
forecasting/
├── lstm_model.py            # побудова та навчання LSTM (TensorFlow/Keras)
├── prophet_model.py         # Prophet-модель для порівняння
├── evaluate.py              # MAE/RMSE/MAPE: LSTM vs Prophet vs naive (--dataset)
├── architecture_study.py    # порівняння конфігурацій LSTM на кількох seed
├── models/
│   ├── lstm_<dataset>.keras       # модель LSTM (перейменований lstm_best.keras)
│   └── prophet_<dataset>.json     # кеш Prophet
├── results/
│   ├── metrics_<dataset>.csv
│   ├── architecture_study_alibaba.csv
│   └── architecture_study_alibaba.png
└── README.md
```

`<dataset>` — `synthetic` або `alibaba`. Артефакти кожного набору мають
власні імена, щоб не змішувати набори: `evaluate.py` відновлює тестову
вибірку з CSV і застосовує scaler, підібраний на цьому ж CSV, тому модель,
навчена на одному наборі й оцінена на іншому, дає хибні метрики.

Покроковий порядок запуску всього проєкту — у гайді «Повний покроковий
гайд». Нижче — лише команди цього модуля.

## Запуск (з теки forecasting/)

Синтетика:

```bash
cp sequences_synthetic.npz sequences.npz
python lstm_model.py --epochs 50 --patience 15 2>&1 | tee step_synthetic.txt
mv -f models/lstm_best.keras models/lstm_synthetic.keras
python evaluate.py --dataset synthetic
```

Alibaba (джерело таблиці результатів — п'ять запусків на конфігурацію):

```bash
PYTHONUTF8=1 python architecture_study.py 2>&1 | tee step_architecture_study.txt
```

Швидка перевірка коду без повного навчання:

```bash
python lstm_model.py --quick                          # 2000 прикладів, 3 епохи
python lstm_model.py --epochs 10                      # обмежена кількість епох
python evaluate.py --dataset synthetic --skip-prophet # без Prophet
python evaluate.py --dataset synthetic --retrain-prophet # ігнорувати кеш Prophet
```

`python lstm_model.py` перезаписує `models/lstm_best.keras`, тому після
кожного навчання перейменовуйте файл (`mv`), як показано вище.

## Архітектура LSTM

```
Input(60, 4) -> LSTM(64, return_sequences=True) -> Dropout(0.2)
              -> LSTM(32) -> Dropout(0.2)
              -> Dense(16, relu) -> Dense(1)
```

- `window_size=60` — модель бачить останню годину навантаження
- `horizon=15` — прогноз на 15 хвилин уперед
- Одна конфігурація `LSTM(64, 32)` для обох наборів даних. Припущення, що
  менша `LSTM(32, 16)` краще на малому реальному трейсі (~8000 навчальних
  прикладів), перевірено `architecture_study.py` на п'яти seed і **не
  підтвердилось**: (64, 32) трохи краща за валідаційною втратою
  (0.00827 ± 0.00017 проти 0.00880 ± 0.00038) і за MAE на тесті
  (3.895 проти 3.996). Різниця невелика й на межі значущості, тож
  коректний висновок — відсутність переваги в меншої конфігурації, а не
  доведена перевага більшої. Конфігурацію обрано за валідаційною втратою,
  тестова вибірка використана лише для звітування.
- EarlyStopping (`--patience`, у наших запусках 15) з відновленням ваг
  найкращої епохи та ReduceLROnPlateau — захист від перенавчання
- `--units U1 U2` задає розмір шарів вручну (так `architecture_study.py`
  порівнює конфігурації)

## Дослідження архітектури (architecture_study.py)

Навчає кілька конфігурацій на кількох seed тим самим протоколом, що й
основне навчання, і оцінює їх тією самою `compare_models` з `evaluate.py`.
Навчання нейромережі стохастичне, і розкид між запусками однієї
конфігурації співмірний із різницею між конфігураціями, тому порівняння на
одному запуску ненадійне.

```bash
python architecture_study.py                           # alibaba, 32,16 і 64,32, 5 seed
python architecture_study.py --seeds 3 --configs 16,8 32,16 64,32
python architecture_study.py --dataset synthetic --configs 64,32 --seeds 5   # ~35 хв
```

Скрипт пише лише `results/architecture_study_<dataset>.csv` і `.png`, а
`models/` та `metrics_*.csv` не чіпає. Повний прогін для Alibaba займає
близько 23 хв.

## Кешування моделей

- `models/lstm_best.keras` створює `ModelCheckpoint` під час навчання;
  після навчання його перейменовують у `models/lstm_<dataset>.keras`, звідки
  його бере `evaluate.py`
- `models/prophet_<dataset>.json` — серіалізована Prophet-модель (JSON, а не
  pickle: стабільніше між версіями CmdStan/Prophet). Якщо файл є,
  `evaluate.py` завантажує його замість повторного навчання

## Порядок імпортів (Windows)

На Windows `import sklearn` **до** `import tensorflow` у тому самому процесі
завершується помилкою `DLL load failed ... 0x45A`; зворотний порядок
працює. Тому в кожному скрипті, де є і TensorFlow, і scikit-learn (зокрема
через `from preprocessing import ...`), спершу має стояти `import
tensorflow`. У `evaluate.py` та `architecture_study.py` це враховано.

## Встановлення Prophet на Windows

Prophet використовує CmdStan. Надійний шлях — версія **1.4.0**: у 1.1.5
вбудована в pip-пакет копія CmdStan неповна на Windows
(`AttributeError: 'Prophet' object has no attribute 'stan_backend'`), а
перевстановлення тієї самої версії проблему не вирішує.

```powershell
python -m pip install --upgrade cmdstanpy
python -m cmdstanpy.install_cxx_toolchain
python -m cmdstanpy.install_cmdstan --compiler

python -m pip uninstall prophet -y
python -m pip install prophet==1.4.0 --no-cache-dir
```

`evaluate.py` не падає, якщо Prophet недоступний: виводить попередження й
показує LSTM проти naive (`--skip-prophet` робить це свідомо).

## Результати

Синтетичний ряд (один запуск; seed навчання не фіксовано — у попередньому
запуску тієї самої конфігурації MAE була 3.706):

| Модель | MAE | RMSE | MAPE, % |
|---|---|---|---|
| LSTM (64, 32) | 3.562 | 4.863 | 14.43 |
| Naive persistence | 3.686 | 5.252 | 16.35 |
| Prophet | 4.708 | 5.863 | 49.34 |

Alibaba Cluster Trace 2018 (середнє ± стандартне відхилення за п'ятьма
запусками, seed 0–4):

| Модель | MAE | RMSE | MAPE, % |
|---|---|---|---|
| LSTM (64, 32) | 3.895 ± 0.080 | 5.134 ± 0.109 | 7.412 ± 0.204 |
| LSTM (32, 16) | 3.996 ± 0.052 | 5.232 ± 0.058 | 7.506 ± 0.065 |
| Naive persistence | 4.519 | 5.839 | 8.47 |

Обидві конфігурації стабільно кращі за naive (навіть найгірший із десяти
запусків, MAE 4.066, краще за 4.519). На синтетиці перевага над naive за
MAE невелика й співмірна з розкидом між запусками; за RMSE та MAPE вона
більша.

Prophet оцінено лише на синтетиці: тижнева сезонність не оцінюється на ~8
добах Alibaba. Його висока MAPE — наслідок методологічної асиметрії:
Prophet прогнозує весь тестовий період одразу від точки навчання
(multi-step-ahead), а LSTM щокроку отримує актуальну історію (rolling
forecast). Деталі — у docstring `evaluate.py::evaluate_prophet()`.

## Наступний крок — Модуль 3 (scaling_logic)

Логіка перетворення прогнозу на кількість реплік Kubernetes та реактивний
коректор на теорії черг.