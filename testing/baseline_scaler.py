"""
Модуль 4.1 — Baseline: пороговий (реактивний) скейлер.

Імітує логіку класичного Kubernetes Horizontal Pod Autoscaler: реагує
ЛИШЕ на поточне вимірюване навантаження, без жодного прогнозування.
Це і є "традиційний метод", з яким методика роботи порівнюється згідно
з задачею 6 з розділу "Мета і задачі роботи" кваліфікаційної роботи.

Формула (аналог реального HPA):
    desired_replicas = ceil(current_load / (capacity_per_replica * target_utilization))

target_utilization (типово 0.7) — HPA свідомо тримає репліки
недовантаженими на 30%, щоб мати запас на час, поки нові поди
стартують. Це та сама ідея, що й наш safety_factor у проактивному
рівні, але тут вона компенсує затримку РЕАКЦІЇ (а не похибку прогнозу).
"""

import math
from dataclasses import dataclass
from datetime import datetime


@dataclass
class BaselineConfig:
    capacity_per_replica_rps: float = 50.0
    target_utilization: float = 0.7
    min_replicas: int = 2
    max_replicas: int = 20
    cooldown_seconds: int = 180  # HPA має довший дефолтний stabilization window на scale-down


class BaselineThresholdScaler:
    """Реактивний скейлер: рішення лише на основі ПОТОЧНОГО навантаження."""

    def __init__(self, config: BaselineConfig):
        self.config = config
        self._last_scale_time: datetime | None = None
        self._last_replicas: int | None = None

    def decide(self, current_load_rps: float, current_time: datetime) -> int:
        raw = current_load_rps / (self.config.capacity_per_replica_rps * self.config.target_utilization)
        calculated = max(self.config.min_replicas, min(self.config.max_replicas, math.ceil(raw)))

        if self._last_replicas is None:
            final = calculated
            self._last_scale_time = current_time
        elif calculated == self._last_replicas:
            final = calculated
        elif self._in_cooldown(current_time):
            final = self._last_replicas
        else:
            final = calculated
            self._last_scale_time = current_time

        self._last_replicas = final
        return final

    def _in_cooldown(self, now: datetime) -> bool:
        if self._last_scale_time is None:
            return False
        return (now - self._last_scale_time).total_seconds() < self.config.cooldown_seconds