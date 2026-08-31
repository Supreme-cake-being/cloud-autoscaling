# data_pipeline — Модуль 1: Збір та підготовка даних

## Структура

```
data_pipeline/
├── data_generator.py     # синтетичний генератор навантаження
├── load_alibaba_data.py  # завантаження й конвертація реального Alibaba trace
├── preprocessing.py      # очистка, нормалізація, sliding window
├── eda.py                 # розвідувальний аналіз + графіки
├── data/                   # згенеровані/завантажені дані (csv, npz)
├── plots/                  # графіки EDA (png)
└── README.md
```

## Запуск (Windows, PowerShell або WSL2)

```bash
pip install pandas numpy scikit-learn matplotlib statsmodels

cd data_pipeline
python data_generator.py   # -> data/synthetic_cluster_load.csv
python preprocessing.py    # -> data/sequences.npz (готово для LSTM)
python eda.py               # -> plots/*.png + статистика в консоль
```

## Отримані результати EDA (на синтетичних даних, 30 днів)

- Середнє навантаження CPU: 26.5% (std 15.8)
- **Максимальний приріст CPU за 15 хв: 63.1 в.п.**
- 99-й перцентиль 15-хв приросту: 10.9 в.п.
- Пропусків після очистки: 0

Це і є емпіричний аргумент на захисті: реактивна система з порогом
(CPU > 80%) реагує лише ПІСЛЯ перевищення, а старт нового поду займає
1-5 хв — за цей час навантаження встигає ще зрости на десятки в.п.
(див. `plots/05_spike_growth_rate.png`), що і призводить до порушення
SLA. Звідси випливає потреба у прогнозі на 10-15 хв вперед (Модуль 2,
LSTM).

## Як підключити реальний датасет Alibaba Cluster Trace 2018

Синтетичні дані потрібні для розробки пайплайну й основного
експерименту. Для додаткової валідації в роботі використано й реальний
trace.

### Крок 1 — завантажити лише потрібні архіви

Офіційний `fetchData.sh` тягне 6 архівів (`machine_meta`,
`machine_usage`, `container_meta`, `container_usage`, `batch_task`,
`batch_instance`). Потрібні лише **перші два** — решта про
контейнери/batch-задачі, поза межами теми роботи:

```bash
#!/bin/bash
url='http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces'
mkdir -p data && cd data
wget -c --retry-connrefused --tries=0 --timeout=50 ${url}/machine_meta.tar.gz
wget -c --retry-connrefused --tries=0 --timeout=50 ${url}/machine_usage.tar.gz
tar -xzf machine_meta.tar.gz
tar -xzf machine_usage.tar.gz
```

### Крок 2 — конвертувати у формат пайплайну

`machine_usage.csv` містить ~247 мільйонів рядків — `load_alibaba_data.py`
читає його **двома проходами частинами** (chunked, ~5М рядків/частина),
щоб не впертись у `MemoryError`: перший прохід визначає топ-N
найзавантаженіших машин, другий фільтрує й агрегує лише їх.

```bash
python load_alibaba_data.py   # -> data/alibaba_cluster_load.csv
```

**Чому агрегація по групі машин, а не одна машина:** LSTM прогнозує
СУКУПНЕ навантаження, яке потім конвертується в "скільки реплік
потрібно" (Модуль 3) — саме так це бачить реальний Kubernetes HPA. Одна
фізична машина не масштабується (вона або є, або її немає).

### Крок 3 — підставити у пайплайн

У `preprocessing.py` та `eda.py` є змінна `DATA_SOURCE`:

```python
DATA_SOURCE = "data/alibaba_cluster_load.csv"  # замість synthetic_cluster_load.csv
```

Решта пайплайну (очистка, нормалізація, sliding window) відпрацює без
змін — обидва джерела дають ідентичний формат
(`timestamp, cpu_util, mem_util, rps, net_in`).

**Застереження:** реальний trace охоплює лише ~8 днів (проти 30 днів
синтетики) — LSTM на ньому має суттєво менше train-прикладів
(~8000 проти ~30000), тому в `forecasting/lstm_model.py` архітектура
моделі **автоматично** підбирається легшою під менший обсяг даних (див.
`forecasting/README.md`).

## Наступний крок — Модуль 2

Маючи `data/sequences.npz` (X_train/y_train/X_val/y_val/X_test/y_test у
форматі, готовому для Keras/PyTorch), переходимо до побудови LSTM-моделі
прогнозування та Prophet-моделі для порівняння.
