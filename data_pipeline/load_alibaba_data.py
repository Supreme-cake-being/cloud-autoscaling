"""
Модуль 1.0b — Завантаження та конвертація реального Alibaba Cluster Trace 2018.

Окремий файл від data_generator.py навмисно: тут інша відповідальність
(читання й перетворення зовнішнього датасету, а не генерація з нуля) та
інша логіка агрегації. Обидва файли дають на виході ІДЕНТИЧНИЙ формат
CSV (timestamp, cpu_util, mem_util, rps, net_in), тому preprocessing.py
та eda.py працюють з будь-яким джерелом без змін — вони не знають і не
повинні знати, звідки взялися дані.

ВАЖЛИВО про розмір файлу: machine_usage.csv у trace-v2018 містить
~247 МІЛЬЙОНІВ рядків (~9+ ГБ лише під один float64-стовпець у пам'яті).
Спроба pd.read_csv() без chunksize впаде з MemoryError навіть на
машині з 16 ГБ RAM. Тому читаємо файл ДВОМА проходами частинами
(chunksize), а не одним pd.read_csv():

  Прохід 1: читаємо тільки [machine_id, cpu_util] (не всі 9 колонок),
            рахуємо середній CPU по кожній машині частинами, щоб
            визначити топ-N найзавантаженіших машин.
  Прохід 2: читаємо файл ще раз частинами, але одразу відкидаємо рядки,
            що НЕ належать відібраним топ-N машинам. У пам'яті
            накопичується лише невелика відфільтрована підмножина
            (< 1% від 247М рядків), а не весь датасет.

Передумова: завантажте machine_meta.tar.gz + machine_usage.tar.gz
(інструкція — у README.md) і розпакуйте в data/machine_usage.csv.

Джерело: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018
"""

import pandas as pd

RAW_COLUMNS = [
    "machine_id", "time_stamp", "cpu_util", "mem_util",
    "mem_gps", "mkpi", "net_in", "net_out", "disk_io",
]

CHUNK_SIZE = 5_000_000  # ~5М рядків на частину — комфортно для 8-16 ГБ RAM


def _find_top_machines(raw_path: str, n_machines: int) -> pd.Index:
    """Прохід 1: визначає топ-N машин за середнім CPU, читаючи файл частинами."""
    sum_cpu = pd.Series(dtype="float64")
    count_cpu = pd.Series(dtype="int64")

    reader = pd.read_csv(
        raw_path, header=None, names=RAW_COLUMNS,
        usecols=[0, 2],  # лише machine_id, cpu_util — економимо пам'ять
        dtype={"machine_id": "category", "cpu_util": "float32"},
        chunksize=CHUNK_SIZE,
    )
    for i, chunk in enumerate(reader):
        grouped = chunk.groupby("machine_id", observed=True)["cpu_util"]
        sum_cpu = sum_cpu.add(grouped.sum(), fill_value=0)
        count_cpu = count_cpu.add(grouped.count(), fill_value=0)
        print(f"  [прохід 1] оброблено частину {i + 1} ({(i + 1) * CHUNK_SIZE:,} рядків)")

    avg_cpu = sum_cpu / count_cpu
    return avg_cpu.sort_values(ascending=False).head(n_machines).index


def load_and_aggregate(
    raw_path: str = "data/machine_usage.csv",
    n_machines: int = 30,
    resample_freq: str = "1min",
) -> pd.DataFrame:
    """
    Читає сирий Alibaba machine_usage.csv (247М рядків, тому chunked)
    та агрегує по групі машин.

    Чому агрегація, а не одна машина: LSTM прогнозує СУКУПНЕ навантаження,
    яке потім конвертується в "скільки реплік потрібно" (Модуль 3) — саме
    так це бачить реальний Kubernetes HPA. Одна фізична машина не
    масштабується (вона або є, або її немає), тож не мапиться на задачу
    дипломної роботи.

    Args:
        raw_path: шлях до розпакованого machine_usage.csv
        n_machines: скільки машин з найвищим середнім CPU відібрати
                    (proxy "продуктивного" кластера, а не тестових серверів)
        resample_freq: крок дискретизації після вирівнювання

    Returns:
        DataFrame у форматі [timestamp, cpu_util, mem_util, rps, net_in]
    """
    print("Прохід 1/2: визначення топ-N машин...")
    top_machines = _find_top_machines(raw_path, n_machines)
    top_machines_set = set(top_machines)

    print("Прохід 2/2: фільтрація та агрегація обраних машин...")
    filtered_chunks = []
    reader = pd.read_csv(
        raw_path, header=None, names=RAW_COLUMNS,
        usecols=[0, 1, 2, 3, 6],  # machine_id, time_stamp, cpu_util, mem_util, net_in
        dtype={
            "machine_id": "category",
            "time_stamp": "int32",
            "cpu_util": "float32",
            "mem_util": "float32",
            "net_in": "float32",
        },
        chunksize=CHUNK_SIZE,
    )
    for i, chunk in enumerate(reader):
        matched = chunk[chunk["machine_id"].isin(top_machines_set)]
        if not matched.empty:
            filtered_chunks.append(matched.copy())
        print(f"  [прохід 2] оброблено частину {i + 1}, знайдено {len(matched)} рядків")

    cluster = pd.concat(filtered_chunks, ignore_index=True)
    del filtered_chunks  # звільняємо пам'ять одразу після concat

    cluster["timestamp"] = pd.to_datetime(cluster["time_stamp"], unit="s", origin="2018-01-01")
    cluster = cluster.drop_duplicates(["machine_id", "timestamp"])

    # CPU/RAM — середнє по кластеру (те, що бачить HPA), мережа — сума
    # (сукупний трафік кластера, а не середній на машину)
    agg = cluster.groupby("timestamp").agg(
        cpu_util=("cpu_util", "mean"),
        mem_util=("mem_util", "mean"),
        net_in=("net_in", "sum"),
    ).reset_index()

    agg["rps"] = agg["cpu_util"] * 10 * n_machines  # проксі-метрика: у trace
                                                       # немає прямого RPS

    df = agg.sort_values("timestamp").set_index("timestamp") \
             .resample(resample_freq).mean().interpolate().reset_index()

    return df[["timestamp", "cpu_util", "mem_util", "rps", "net_in"]]


if __name__ == "__main__":
    df = load_and_aggregate()
    out_path = "data/alibaba_cluster_load.csv"
    df.to_csv(out_path, index=False)
    print(f"Оброблено {len(df)} записів -> {out_path}")
    print(df.describe())