# Повний покроковий гайд: від чистого середовища до результатів для розділу 3

*Проєкт cloud-autoscaling · Windows, Python 3.12, Git Bash · кроки виконуються послідовно*

## Коротко

1. Середовище: venv, requirements, перевірка імпортів (п. 1).
2. Дані: синтетика й Alibaba, кожен набір окремо (п. 2).
3. Прогнозування: LSTM навчається й оцінюється окремо на кожному наборі (п. 3).
4. Логіка масштабування: демо-скрипти та перевірки (п. 4).
5. Порівняння з baseline: симуляція (п. 5).
6. Збір результатів для тексту роботи (п. 6).

Три правила, які заощаджують години: кожен скрипт запускайте з теки його модуля (шляхи відносні); у Windows ставте пакети через `python -m pip`, а не `pip`; артефакти кожного набору даних називайте з суфіксом `_synthetic` або `_alibaba` — це захищає від змішування наборів (чому це критично — п. 7).

## 1. Середовище (один раз)

```bash
cd /e/Github/cloud-autoscaling
python -m venv venv
source venv/Scripts/activate        # PowerShell: venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Пакети `kubernetes` і `locust` потрібні лише для необов'язкового живого тесту — їхні рядки в `requirements.txt` можна закоментувати.

Перевірка (кожна команда має завершитись без помилок):

```bash
python -c "import pandas, numpy, sklearn, matplotlib, statsmodels; print('data stack OK')"
python -c "import tensorflow as tf; print('TF', tf.__version__)"
python -c "from prophet import Prophet; Prophet(); print('Prophet OK')"
```

Очікувано: `TF 2.21.0`; повідомлення «Importing plotly failed» нешкідливе. Якщо щось падає — п. 8.

## 2. Модуль 1: дані (data\_pipeline)

**Синтетичний набір** (основний експеримент):

```bash
cd data_pipeline
python data_generator.py
```

Очікувано: «Згенеровано 43200 записів». У `preprocessing.py` (блок `__main__`) має стояти `DATA_SOURCE = "data/synthetic_cluster_load.csv"`. Далі:

```bash
python preprocessing.py
cp data/sequences.npz data/sequences_synthetic.npz
python eda.py
```

Очікувано: Train (30165, 60, 4), Val (6407, 60, 4), Test (6406, 60, 4); EDA: mean\_cpu 26.52, max\_15min\_growth 63.10, p99\_15min\_growth 10.86. Примітка: у версії `eda.py` з цього проєкту ряд генерується всередині скрипта (seed 42), змінної `DATA_SOURCE` там немає — у README модуля це описано неточно.

**Alibaba** (валідація на реальних даних). Передумова — розпакований `data/machine_usage.csv` (див. README модуля):

```bash
python load_alibaba_data.py
```

Скрипт робить два проходи по \~247 млн рядків, це кілька хвилин. Очікувано: `data/alibaba_cluster_load.csv`, близько 11–12 тис. рядків (\~8 діб). Далі змініть у `preprocessing.py` рядок на `DATA_SOURCE = "data/alibaba_cluster_load.csv"`:

```bash
python preprocessing.py
cp data/sequences.npz data/sequences_alibaba.npz
```

Очікувано: Train ≈ (7989, 60, 4), Val ≈ (1655, 60, 4). Якщо Train дорівнює 30165 — ви знову обробили синтетику, перевірте `DATA_SOURCE`.

Передайте послідовності у forecasting:

```bash
cp data/sequences_synthetic.npz data/sequences_alibaba.npz ../forecasting/
```

## 3. Модуль 2: прогнозування (forecasting)

Підготовка (один раз):

- Замініть `forecasting/evaluate.py` оновленою версією: вона має параметр `--dataset` і сама добирає CSV, модель, кеш Prophet та файл результатів під обраний набір (п. 7).
- Рекомендовано: у `lstm_model.py` одразу після імпортів додайте рядок `keras.utils.set_random_seed(42)` — він фіксує випадковість Python, NumPy і TensorFlow. Це зменшує розбіжності між запусками, але не гарантує їх повної відсутності.

Смоук-тест (перезаписує `models/lstm_best.keras`, тому запускайте його до справжнього навчання):

```bash
cd ../forecasting
python lstm_model.py --quick
```

**Синтетика:**

```bash
cp sequences_synthetic.npz sequences.npz
python lstm_model.py --epochs 50 --patience 15 2>&1 | tee step_synthetic.txt
mv -f models/lstm_best.keras models/lstm_synthetic.keras
python evaluate.py --dataset synthetic
```

Очікувано: `LSTM(64, 32)`, зупинка за early stopping (зазвичай 20–30 епох), таблиця з трьох рядків — LSTM, Naive persistence, Prophet. Контроль: для синтетики naive persistence завжди дає 3.686 / 5.252 / 16.35.

**Alibaba:**

```bash
cp sequences_alibaba.npz sequences.npz
python lstm_model.py --epochs 50 --patience 15 2>&1 | tee step_alibaba.txt
mv -f models/lstm_best.keras models/lstm_alibaba.keras
python evaluate.py --dataset alibaba --skip-prophet
```

Очікувано: `LSTM(32, 16)` (навчальних прикладів менше 15000). Контроль: рядок Naive persistence має відрізнятися від 3.686 / 5.252 / 16.35 — якщо збігається, оцінка пішла на синтетичних даних.

Prophet для Alibaba пропущено свідомо: тижнева сезонність не оцінюється на \~8 добах (у навчальній частині менше тижня), тому порівняння LSTM з naive тут доречніше. Якщо Prophet усе ж потрібен — приберіть `--skip-prophet`.

Результати: `results/metrics_synthetic.csv` та `results/metrics_alibaba.csv`. Старий `metrics_comparison.csv` можна видалити.

## 4. Модуль 3: логіка масштабування (scaling\_logic)

```bash
cd ../scaling_logic
python decision_engine.py
python queuing_corrector.py
python k8s_client.py
python autoscaler.py
python sanity_checks.py
```

Очікувано: cooldown дає 5, 5, 5, 4 репліки; коректор — 3, 6, 6; autoscaler: цикл 1 — 3→3, цикл 2 — 3→6 (втрутився коректор); `sanity_checks.py` завершується словами «Усі тести пройшли». TensorFlow для цього модуля не потрібен.

## 5. Модуль 4: порівняння з baseline (testing)

```bash
cd ../testing
python simulate_comparison.py
```

Необов'язково: `--max-delay 15` розширює аналіз чутливості. TensorFlow не потрібен. Очікувано при затримці старту 3 хв: гібрид — 7946 подій масштабування, SLA 95.25 %, вартість $306.78; baseline — 11903, 96.82 %, $298.91. По SLA гібрид випереджає baseline починаючи з 5 хв.

Увага: константа `LSTM_MAE_RPS = 37.0` у `simulate_comparison.py` дорівнює MAE LSTM на синтетиці (≈3.7 в.п.), помноженій на 10. Якщо після перенавчання MAE суттєво змінилась — оновіть константу й перезапустіть симуляцію.

## 6. Що йде в розділ 3

| Матеріал | Звідки | Куди |
| --- | --- | --- |
| Рисунки 3.1–3.5 | `data_pipeline/plots/01…05` | підрозділ 3.2 |
| Таблиця зі статистикою ряду | вивід `eda.py` | 3.2 |
| Таблиця розмірів вибірок | вивід `preprocessing.py` (обидва набори) | 3.2 |
| Метрики LSTM / Prophet / naive | `forecasting/results/metrics_synthetic.csv` | 3.3 |
| Метрики на реальних даних | `forecasting/results/metrics_alibaba.csv` | 3.3 |
| Порівняння з baseline | `testing/results/comparison_metrics.csv`, `comparison_chart.png` | 3.5 |
| Аналіз чутливості | `testing/results/sensitivity_analysis.csv`, `.png` | 3.5 |

## 7. Чому набори даних не можна змішувати, і що з відтворюваністю

- `evaluate.py` відновлює тестову вибірку з CSV і застосовує scaler, підібраний на цьому ж CSV. Модель, навчена на Alibaba й оцінена на синтетичному CSV, дає хибні метрики. Саме тому `--dataset` вибирає одразу CSV, модель, кеш Prophet і файл результатів.
- Дані (seed 42) і симуляція детерміновані. Навчання LSTM — ні: навіть із фіксованим seed між машинами можливі невеликі розбіжності. Фінальне навчання зробіть один раз, збережіть `models/*.keras`, `results/*.csv` і логи `step_*.txt`, а перед заповненням таблиць не перенавчайте.
- `.gitignore` виключає `models/` і `results/`. Артефакти, що йдуть у текст, збережіть окремо або додайте через `git add -f`.

## 8. Типові проблеми

| Симптом | Причина | Рішення |
| --- | --- | --- |
| `bash: .../pip: Permission denied` | `pip.exe` у venv заблоковано | `python -m pip ...` |
| `ModuleNotFoundError: config` або `preprocessing` | скрипт запущено не з теки модуля | `cd` у теку модуля |
| Prophet: `no attribute 'stan_backend'` | у prophet 1.1.5 вбудована копія CmdStan неповна на Windows | `prophet==1.4.0`; за потреби `python -m cmdstanpy.install_cxx_toolchain` і `python -m cmdstanpy.install_cmdstan --compiler`; обхід — `--skip-prophet` |
| Живий тест (Docker, kubectl) не піднімається | не потрібен для результатів роботи | див. `testing/LIVE_TEST_GUIDE.md` і статус у `testing/README.md` |

**TensorFlow: `DLL load failed ... 0x45A`.** Типовий випадок: падає `evaluate.py`, хоча «голий» `import tensorflow` працює. Ймовірна причина — порядок імпортів: numpy/pandas/scikit-learn завантажуються до TensorFlow. Причину ще підтверджуємо, тож почніть з діагностики — вона покаже, який саме імпорт провокує збій:

```bash
python -c "import numpy; import tensorflow; print('numpy->tf OK')"
python -c "import pandas; import tensorflow; print('pandas->tf OK')"
python -c "import sklearn.metrics; import tensorflow; print('sklearn->tf OK')"
python -m pip check
```

Оновлений `evaluate.py` імпортує TensorFlow першим. Якщо він усе одно падає — запустіть із `export KMP_DUPLICATE_LIB_OK=TRUE`. Якщо падає й `lstm_model.py` — надішліть вивід чотирьох команд вище. Якщо не працює навіть «голий» імпорт:

```bash
python -m pip install tensorflow==2.21.0 numpy==1.26.4 protobuf --force-reinstall --no-deps
```

## Що надіслати на перевірку

- `forecasting/results/metrics_synthetic.csv` і `metrics_alibaba.csv`
- `step_synthetic.txt` і `step_alibaba.txt` (логи навчання)
- `data_pipeline/data/alibaba_cluster_load.csv` або хоча б кількість його рядків і перші 5 рядків
