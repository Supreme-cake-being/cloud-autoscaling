"""
Модуль 3.4 — Orchestrator: об'єднання проактивного + реактивного рівнів.

Це і є "гібридна методика" з теми роботи в коді: на кожному циклі
контролер (1) бере прогноз LSTM і рахує проактивне рішення, (2) бере
поточні РЕАЛЬНІ метрики і пропускає через реактивний коректор, (3)
фінальне число реплік = максимум з двох (коректор ніколи не зменшує
те, що вирішив проактивний рівень — лише піднімає вище, якщо поточна
черга того вимагає), (4) застосовує через K8sScaler.

У реальному розгортанні (Модуль 4) проактивний цикл запускається рідше
(раз на 1-5 хв, синхронно з новим прогнозом LSTM), а реактивний —
частіше (кожні 10-15с, на кожному новому знятті метрик з Prometheus).
Тут для простоти демонстрації обидва виконуються в одному циклі.
"""

from dataclasses import dataclass
from datetime import datetime

from config import ScalingConfig
from decision_engine import ProactiveDecisionEngine
from queuing_corrector import ReactiveCorrector
from k8s_client import K8sScaler


@dataclass
class AutoscalingCycleResult:
    proactive_replicas: int
    final_replicas: int
    reactive_triggered: bool
    proactive_reason: str
    timestamp: datetime


class HybridAutoscaler:
    def __init__(self, config: ScalingConfig, dry_run: bool = True):
        self.config = config
        self.proactive = ProactiveDecisionEngine(config)
        self.reactive = ReactiveCorrector(config)
        self.k8s = K8sScaler(config.deployment_name, config.namespace, dry_run=dry_run)

    def run_cycle(
        self, predicted_load_rps: float, current_actual_rps: float, current_time: datetime | None = None,
    ) -> AutoscalingCycleResult:
        """
        Один цикл прийняття рішення.

        Args:
            predicted_load_rps: прогноз LSTM на найближчі horizon хвилин
            current_actual_rps: реальний поточний RPS (з Prometheus/Locust)
        """
        proactive_decision = self.proactive.decide(predicted_load_rps, current_time)
        correction = self.reactive.correct(current_actual_rps, proactive_decision.replicas)

        self.k8s.scale(correction.replicas)

        return AutoscalingCycleResult(
            proactive_replicas=proactive_decision.replicas,
            final_replicas=correction.replicas,
            reactive_triggered=correction.triggered,
            proactive_reason=proactive_decision.reason,
            timestamp=proactive_decision.timestamp,
        )


if __name__ == "__main__":
    config = ScalingConfig(cooldown_seconds=60)
    autoscaler = HybridAutoscaler(config, dry_run=True)

    # Сценарій: прогноз каже "усе спокійно", але реальність — раптовий сплеск
    print("=== Сценарій: прогноз пропустив сплеск, реактивний коректор рятує SLA ===\n")

    print("--- Цикл 1: прогноз і реальність збігаються (нормальне навантаження) ---")
    r1 = autoscaler.run_cycle(predicted_load_rps=100, current_actual_rps=105)
    print(f"Проактивно: {r1.proactive_replicas} | Фінально: {r1.final_replicas} | "
          f"Коректор втручався: {r1.reactive_triggered}\n")

    print("--- Цикл 2: раптовий сплеск, якого LSTM не передбачила ---")
    r2 = autoscaler.run_cycle(predicted_load_rps=110, current_actual_rps=290)
    print(f"Проактивно: {r2.proactive_replicas} | Фінально: {r2.final_replicas} | "
          f"Коректор втручався: {r2.reactive_triggered}")
    print(f"({r2.proactive_reason})")