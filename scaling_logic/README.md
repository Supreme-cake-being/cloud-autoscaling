# scaling_logic — Модуль 3: Логіка масштабування

## Структура

```
scaling_logic/
├── config.py               # усі пороги/параметри в одному місці
├── decision_engine.py       # проактивний рівень: прогноз -> репліки, cooldown
├── queuing_corrector.py     # реактивний рівень: теорія черг (M/M/c, Erlang C)
├── k8s_client.py             # інтеграція з Kubernetes API (dry-run за замовчуванням)
├── autoscaler.py             # orchestrator: об'єднує все в один цикл
└── README.md
```

## Запуск (кожен файл можна тестувати ізольовано, без K8s-кластера)

```bash
cd scaling_logic
python decision_engine.py      # демо cooldown-механізму
python queuing_corrector.py    # демо реактивного коректора на сплеску
python k8s_client.py            # демо dry-run режиму
python autoscaler.py            # демо повного гібридного циклу
```

Усе працює **без підключеного Kubernetes-кластера** — `K8sScaler` за
замовчуванням у dry-run режимі (лише друкує дію, не звертається до
кластера). Коли підключите Minikube/Docker Desktop K8s/реальний
кластер — просто передайте `dry_run=False`.

## Архітектура рішення

**Проактивний рівень** (`decision_engine.py`):

```
replicas = clamp(ceil(predicted_load * safety_factor / capacity_per_replica),
                  min_replicas, max_replicas)
```

З cooldown-захистом від flapping (тремтіння) — після зміни кількості
реплік наступні `cooldown_seconds` секунд повертається попереднє
значення, навіть якщо новий розрахунок інший.

**Реактивний коректор** (`queuing_corrector.py`):
Теорія черг M/M/c + формула Erlang C — рахує мінімальну кількість
серверів, при якій очікуваний час очікування в черзі не перевищує SLA.
Використовує ПОТОЧНІ (реальні), а не прогнозовані метрики — ловить
сплески, які прогноз пропустив. Ніколи не зменшує рішення проактивного
рівня, лише піднімає вище за потреби.

**Фінальне рішення** = `max(proactive, reactive)`.

## Підключення реального прогнозу з Модуля 2

Зараз `autoscaler.py` демонструється на вручну заданих числах
(`predicted_load_rps`, `current_actual_rps`). Для реальної інтеграції:

```python
import sys
sys.path.insert(0, "../forecasting")
import tensorflow as tf

model = tf.keras.models.load_model("../forecasting/models/lstm_best.keras")
# ... отримати останні 60 хв метрик, прогнати через model.predict(),
# денормалізувати через scaler.inverse_transform() (див. Модуль 2,
# evaluate.py::compare_models для прикладу денормалізації)
predicted_rps = ...

autoscaler.run_cycle(predicted_load_rps=predicted_rps, current_actual_rps=live_rps_from_prometheus)
```

Повна інтеграція з живими метриками (Prometheus) та тестовим
навантаженням (Locust) — Модуль 4.

## Підбір параметрів під ваш тестовий сервіс

`capacity_per_replica_rps`, `service_rate_per_replica` та
`max_wait_time_sec` у `config.py` зараз орієнтовні. Коли розгорнете
тестовий сервіс під Locust (Модуль 4), варто відкалібрувати ці числа
під реальні виміри пропускної здатності одного пода — інакше
розрахунки коректні математично, але не відображають реальну ємність
вашого конкретного сервісу.

## Наступний крок — Модуль 4 (testing)

Генерація реалістичного навантаження через Locust, порівняння з
baseline (стандартний Kubernetes HPA), метрики: latency P95/P99,
порушення SLA, утилізація CPU, вартість ресурсів.
