# Модуль 1 — Збір та підготовка даних

## Структура

```
module1_data_pipeline/
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
(див. `plots/05_spike_growth_rate.png`), що і призводить до порушення SLA.
Звідси випливає потреба у прогнозі на 10-15 хв вперед (Модуль 2, LSTM).

## Як підключити реальний датасет Alibaba Cluster Trace 2018

Синтетичні дані потрібні лише для розробки пайплайну. Для фінальної
валідації в роботі варто додати реальний trace.

### Крок 1 — завантажити лише потрібні архіви

Офіційний `fetchData.sh` тягне 6 архівів (`machine_meta`,
`machine_usage`, `container_meta`, `container_usage`, `batch_task`,
`batch_instance`). Нам потрібні лише **перші два** — решта це про
контейнери/batch-задачі, що поза межами нашої теми, а `container_usage`
до того ж займає десятки ГБ:

```bash
#!/bin/bash
url='http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces'
mkdir -p data && cd data
wget -c --retry-connrefused --tries=0 --timeout=50 ${url}/machine_meta.tar.gz
wget -c --retry-connrefused --tries=0 --timeout=50 ${url}/machine_usage.tar.gz
tar -xzf machine_meta.tar.gz
tar -xzf machine_usage.tar.gz
```

`machine_usage.csv` — це і є прямий аналог нашого синтетичного датасету:
часовий ряд `cpu_util_percent` / `mem_util_percent` кожні ~300 сек по
кожній машині кластера. `machine_meta.csv` дає контекст (`cpu_num`,
`mem_size` кожної машини — знадобиться, якщо захочете нормувати
навантаження відносно ємності машини).

### Крок 2 — конвертувати у формат пайплайну

Логіка конвертації (парсинг схеми, агрегація по топ-N машинах, resample
до рівномірного кроку) винесена в окремий файл **`load_alibaba_data.py`**
— не в `data_generator.py`, оскільки це інша відповідальність (читання
й перетворення зовнішнього датасету, а не генерація з нуля):

```bash
python load_alibaba_data.py   # -> data/alibaba_cluster_load.csv
```

### Крок 3 — підставити у пайплайн

У `preprocessing.py` та `eda.py` є змінна `DATA_SOURCE` — просто
замініть шлях:

```python
DATA_SOURCE = "data/alibaba_cluster_load.csv"  # замість synthetic_cluster_load.csv
```

Решта пайплайну (очистка, нормалізація, sliding window) відпрацює без
змін — `data_generator.py` і `load_alibaba_data.py` дають ідентичний
формат виходу (`timestamp, cpu_util, mem_util, rps, net_in`).

## Наступний крок — Модуль 2

Маючи `data/sequences.npz` (X_train/y_train/X_val/y_val/X_test/y_test у
форматі, готовому для Keras/PyTorch), переходимо до побудови LSTM-моделі
прогнозування та Prophet-моделі для порівняння.
