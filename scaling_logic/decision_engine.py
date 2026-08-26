"""
Модуль 3.1 — Проактивний рівень: прогноз навантаження -> кількість реплік.

Архітектурне рішення:

Формула розрахунку:
    replicas = ceil(predicted_load * safety_factor / capacity_per_replica)

Чому саме так:
    - Ділимо прогнозоване навантаження на ємність однієї репліки —
      базова логіка будь-якого capacity planning.
    - safety_factor (>1.0) закладає запас: LSTM має похибку (MAE з
      Модуля 2, ~3-4 percentage points на наших даних), і потрібен час
      на "розігрів" нового поду (readiness probe) — краще трохи
      "переоцінити" потребу, ніж постійно балансувати на межі SLA.
    - ceil(), а не round() — округлення ВГОРУ: нестача реплік означає
      порушення SLA, надлишок — лише трохи зайвих витрат. Асиметричний
      ризик, тому й округлення асиметричне.

Cooldown-механізм:
    Без нього прогноз, що коливається біля порогу переходу (напр. між
    4 і 5 репліками), спричиняє "тремтіння" (flapping) — скейлінг щохвилини
    туди-сюди, що саме по собі дестабілізує систему (постійний запуск/
    зупинка подів). Рішення: після кожної ЗМІНИ кількості реплік
    "заморожуємо" наступні рішення на cooldown_seconds, повертаючи
    попереднє значення, навіть якщо новий розрахунок відрізняється.
"""

import math
from dataclasses import dataclass
from datetime import datetime

from config import ScalingConfig


@dataclass
class ScalingDecision:
    replicas: int
    reason: str
    timestamp: datetime
    raw_calculated: int  # значення ДО cooldown-обмеження, для логування/аналізу


class ProactiveDecisionEngine:
    """
    Стейтфул-об'єкт: пам'ятає час і результат останнього акту
    масштабування, щоб реалізувати cooldown. У реальному розгортанні
    (Модуль 4) створюється один екземпляр на весь цикл роботи
    контролера, а не по одному на кожен виклик.
    """

    def __init__(self, config: ScalingConfig):
        self.config = config
        self._last_scale_time: datetime | None = None
        self._last_replicas: int | None = None

    def decide(self, predicted_load_rps: float, current_time: datetime | None = None) -> ScalingDecision:
        """
        Args:
            predicted_load_rps: прогнозоване навантаження (RPS) від LSTM
                                 (вже денормалізоване зі шкали [0,1] назад
                                 у реальні одиниці — див. Модуль 2, scaler)
            current_time: час прийняття рішення (параметр для тестованості
                          — у продакшн-циклі не передається, використовує
                          datetime.utcnow())
        """
        current_time = current_time or datetime.utcnow()

        raw = predicted_load_rps * self.config.safety_factor / self.config.capacity_per_replica_rps
        calculated = max(self.config.min_replicas, min(self.config.max_replicas, math.ceil(raw)))

        if self._last_replicas is None:
            # Перший виклик — cooldown ще не застосовується
            final = calculated
            reason = f"перший розрахунок: прогноз={predicted_load_rps:.1f} RPS -> {calculated} реплік"
            self._last_scale_time = current_time
        elif calculated == self._last_replicas:
            # Нічого не змінюється — cooldown не потрібен, просто підтверджуємо
            final = calculated
            reason = "без змін"
        elif self._in_cooldown(current_time):
            # Розрахунок хоче змінити кількість реплік, але ще діє cooldown
            # після попередньої зміни — тримаємо попереднє значення
            final = self._last_replicas
            remaining = self.config.cooldown_seconds - (current_time - self._last_scale_time).total_seconds()
            reason = f"cooldown ще {remaining:.0f}с — тримаємо {final} (розрахунок хотів {calculated})"
        else:
            # Cooldown минув, розрахунок відрізняється — застосовуємо нове значення
            final = calculated
            reason = f"масштабування {self._last_replicas} -> {final} (прогноз={predicted_load_rps:.1f} RPS)"
            self._last_scale_time = current_time

        self._last_replicas = final
        return ScalingDecision(replicas=final, reason=reason, timestamp=current_time, raw_calculated=calculated)

    def _in_cooldown(self, now: datetime) -> bool:
        if self._last_scale_time is None:
            return False
        return (now - self._last_scale_time).total_seconds() < self.config.cooldown_seconds


if __name__ == "__main__":
    from datetime import timedelta

    config = ScalingConfig(cooldown_seconds=120)
    engine = ProactiveDecisionEngine(config)

    # Симуляція: навантаження коливається біля межі переходу 4->5 реплік
    t0 = datetime(2025, 1, 1, 12, 0, 0)
    scenario = [
        (t0, 180),                              # ceil(180*1.15/50)=5
        (t0 + timedelta(seconds=30), 170),      # ceil(170*1.15/50)=4 -> у cooldown
        (t0 + timedelta(seconds=60), 190),      # ще у cooldown
        (t0 + timedelta(seconds=130), 170),     # cooldown минув -> застосує 4
    ]

    print("Демонстрація cooldown-механізму (боротьба з flapping):\n")
    for t, load in scenario:
        decision = engine.decide(load, current_time=t)
        print(f"[{t.strftime('%H:%M:%S')}] прогноз={load} RPS -> replicas={decision.replicas} | {decision.reason}")