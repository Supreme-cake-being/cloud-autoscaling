"""
Модуль 4.2 — Симулятор флоту подів із затримкою старту.

Найважливіший елемент усього порівняння: команда "масштабувати до N
реплік" і момент, коли ці репліки РЕАЛЬНО почнуть обробляти трафік —
це не одна й та сама мить. Реальний под потребує 1-5 хв на старт
(pull образу, ініціалізація застосунку, readiness probe).

Саме ця затримка — і є та причина, чому проактивний підхід кращий:
- Реактивний скейлер: бачить перевантаження ЗАРАЗ -> віддає команду
  ЗАРАЗ -> под стає активним через startup_delay -> весь цей час
  система обслуговує запити СТАРОЮ (недостатньою) кількістю реплік.
- Проактивний скейлер: бачить прогноз на horizon хвилин вперед ->
  віддає команду ЗАРАЗ (заздалегідь) -> под встигає стати активним
  ДО того, як навантаження реально прийде.

Симулятор явно розділяє "скільки реплік замовлено" (commanded) від
"скільки реально активні й обробляють трафік" (active) — усі метрики
ефективності (CPU utilization, latency, SLA) рахуються по active,
а не по commanded.
"""

from dataclasses import dataclass, field


@dataclass
class PodFleetSimulator:
    startup_delay_steps: int  # скільки кроків симуляції новий под "розігрівається"
    active_replicas: int = 2

    # Черга подів, що ще стартують: список (крок_коли_стане_активним, кількість)
    _pending: list = field(default_factory=list)

    def step(self, commanded_replicas: int, current_step: int) -> int:
        """
        Викликається на кожному кроці симуляції.

        Args:
            commanded_replicas: скільки реплік ЗАМОВЛЕНО скейлером на цьому кроці
            current_step: індекс поточного кроку симуляції

        Returns:
            active_replicas: скільки реплік реально активні ПІСЛЯ цього кроку
        """
        # Активуємо поди, чий час "розігріву" вже настав
        still_pending = []
        for ready_step, count in self._pending:
            if ready_step <= current_step:
                self.active_replicas += count
            else:
                still_pending.append((ready_step, count))
        self._pending = still_pending

        diff = commanded_replicas - (self.active_replicas + sum(c for _, c in self._pending))

        if diff > 0:
            # Масштабування ВГОРУ: нові поди йдуть у чергу на "розігрів",
            # активними стають лише через startup_delay_steps
            self._pending.append((current_step + self.startup_delay_steps, diff))
        elif diff < 0:
            # Масштабування ВНИЗ: у K8s термінація подів практично
            # миттєва (на відміну від старту) — зменшуємо одразу
            self.active_replicas = max(0, self.active_replicas + diff)

        return self.active_replicas