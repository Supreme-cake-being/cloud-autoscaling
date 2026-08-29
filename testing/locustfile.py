"""
Модуль 4.4 — Locust-сценарій для живого навантажувального тестування.

Використовується ТІЛЬКИ коли є підключений Kubernetes-кластер (Minikube/
Docker Desktop K8s/реальний) з розгорнутим тестовим сервісом. До того
часу для отримання метрик ефективності використовуйте
simulate_comparison.py (не потребує інфраструктури).

Запуск (після розгортання тестового сервісу через k8s/deployment.yaml):
    locust -f locustfile.py --host http://<service-external-ip>

Веб-інтерфейс Locust (http://localhost:8089) дозволяє задати кількість
користувачів і темп їх приросту вручну — корисно для відтворення
патернів навантаження зі синтетичного датасету (Модуль 1) наживо.
"""

from locust import HttpUser, task, between


class CloudServiceUser(HttpUser):
    # Пауза між запитами одного "користувача" — імітує реальний трафік,
    # а не суцільний потік запитів на максимальній швидкості
    wait_time = between(0.5, 2.0)

    @task(3)
    def get_light_endpoint(self):
        """Легкий запит (напр. health-check чи статична сторінка)."""
        self.client.get("/health")

    @task(1)
    def get_heavy_endpoint(self):
        """
        Важчий запит — імітує реальне навантаження на CPU (обробка
        даних, рендеринг тощо). Співвідношення 3:1 з легким запитом —
        реалістичний профіль трафіку більшості вебсервісів.
        """
        self.client.get("/compute")


# Для відтворення конкретного патерну навантаження (напр. з
# synthetic_cluster_load.csv) замість ручного керування через веб-UI,
# використовуйте LoadTestShape:
#
# from locust import LoadTestShape
#
# class ReplayHistoricalLoad(LoadTestShape):
#     """Відтворює навантаження з CSV-файлу секунда-в-секунду."""
#     def __init__(self):
#         import pandas as pd
#         self.load_curve = pd.read_csv(
#             "../data_pipeline/data/synthetic_cluster_load.csv"
#         )["rps"].tolist()
#
#     def tick(self):
#         run_time = int(self.get_run_time())
#         if run_time >= len(self.load_curve):
#             return None
#         target_rps = self.load_curve[run_time]
#         users = int(target_rps / 2)  # приблизне співвідношення users->RPS,
#                                        # підбирається під ваш конкретний сервіс
#         return (users, users)  # (кількість users, spawn rate)