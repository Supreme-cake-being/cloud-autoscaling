"""
Модуль 3.2 — Реактивний коректор на основі теорії черг (M/M/c, Erlang C).

Навіщо потрібен окремо від проактивного рівня:
    LSTM (Модуль 2) прогнозує на 15 хв вперед — але прогноз ніколи не
    ідеальний, і НЕПЕРЕДБАЧЕНІ сплески (flash-crowd, збій в іншому
    сервісі, що перенаправив трафік сюди) прогноз просто не побачить
    заздалегідь. Реактивний коректор працює з ПОТОЧНИМИ (не
    прогнозованими) метриками в реальному часі й миттєво додає
    репліки, якщо система наближається до порушення SLA — це і є
    "гібридність" методики з теми роботи.

Математична модель — M/M/c черга:
    M/M/c — стандартна модель теорії масового обслуговування:
    - M (Markovian) — надходження запитів має пуассонівський розподіл
      (типове й обґрунтоване припущення для вебтрафіку)
    - M — час обслуговування експоненційно розподілений
    - c — кількість серверів (реплік), що обслуговують чергу СПІЛЬНО

    Formula Erlang C дає ймовірність того, що запит потрапить у чергу
    (а не обслужиться миттєво), звідки виводиться очікуваний час
    очікування E[W]. Задача: знайти МІНІМАЛЬНЕ c, при якому E[W] не
    перевищує SLA-поріг (max_wait_time_sec).

Чому не просто "utilization > 80% -> додати репліку" (як типовий HPA):
    Порогове правило не враховує НЕЛІНІЙНІСТЬ черг — при utilization
    ближче до 100% час очікування зростає не лінійно, а гіперболічно
    (черга "вибухає"). M/M/c явно моделює цю нелінійність і дає
    математично обґрунтовану, а не евристичну, кількість серверів.
"""

import math
from dataclasses import dataclass

from config import ScalingConfig


def erlang_c_wait_probability(c: int, rho: float) -> float:
    """
    Ймовірність Erlang C: що новий запит застане всі c серверів зайнятими
    і потрапить у чергу (замість миттєвого обслуговування).

    Args:
        c: кількість серверів (реплік)
        rho: завантаженість системи (0 < rho < 1) = a / c, де
             a = arrival_rate / service_rate (запропоноване навантаження
             в Ерлангах)
    """
    a = rho * c
    sum_terms = sum((a ** k) / math.factorial(k) for k in range(c))
    last_term = (a ** c) / (math.factorial(c) * (1 - rho))
    return last_term / (sum_terms + last_term)


def expected_wait_time(c: int, arrival_rate: float, service_rate: float) -> float:
    """
    Очікуваний час очікування запиту в черзі (секунди) за формулою
    Erlang C: E[W] = P(wait) / (c*mu - lambda).

    Повертає float('inf'), якщо система нестабільна (arrival_rate >=
    c * service_rate — черга росте необмежено, будь-який скінченний
    час очікування недосяжний).
    """
    rho = arrival_rate / (c * service_rate)
    if rho >= 1:
        return float("inf")
    p_wait = erlang_c_wait_probability(c, rho)
    return p_wait / (c * service_rate - arrival_rate)


def min_servers_for_sla(
    arrival_rate: float, service_rate: float, max_wait: float, max_c: int = 200,
) -> int:
    """
    Мінімальна кількість серверів (реплік), при якій очікуваний час
    очікування в черзі не перевищує max_wait.

    Лінійний пошук від точки стабільності (arrival_rate/service_rate)
    вгору — для реалістичних діапазонів c (<200) це швидко (<1мс),
    складніші методи (бінарний пошук) не виправдані для такого масштабу.
    """
    c = max(1, math.ceil(arrival_rate / service_rate))  # мінімум для стабільності системи
    while c <= max_c:
        if expected_wait_time(c, arrival_rate, service_rate) <= max_wait:
            return c
        c += 1
    return max_c  # система не тягне навіть при максимумі — повертаємо стелю


@dataclass
class CorrectionResult:
    replicas: int
    triggered: bool  # чи коректор реально втрутився (перевищив проактивне рішення)
    current_wait_estimate: float


class ReactiveCorrector:
    """
    Використовує ПОТОЧНІ метрики (не прогноз) для миттєвої корекції.
    У продакшн-циклі викликається значно частіше за проактивний рівень
    (напр. кожні 10-15с проти кожні 1-5 хв для прогнозу) — саме тому
    він і "реактивний": ловить те, що проактивний рівень пропустив
    між своїми циклами прийняття рішень.
    """

    def __init__(self, config: ScalingConfig):
        self.config = config

    def correct(self, current_arrival_rate_rps: float, proactive_replicas: int) -> CorrectionResult:
        """
        Args:
            current_arrival_rate_rps: РЕАЛЬНИЙ поточний RPS (з Prometheus/
                                       Locust у Модулі 4, а не прогноз)
            proactive_replicas: кількість реплік, вже призначена
                                 проактивним рівнем (Модуль 3.1) —
                                 коректор ніколи не ЗМЕНШУЄ це значення,
                                 лише додає репліки понад нього, якщо
                                 поточна черга вимагає більше
        """
        needed = min_servers_for_sla(
            current_arrival_rate_rps,
            self.config.service_rate_per_replica,
            self.config.max_wait_time_sec,
        )
        needed = min(needed, self.config.max_replicas)

        final = max(proactive_replicas, needed)
        wait_estimate = expected_wait_time(
            max(proactive_replicas, 1), current_arrival_rate_rps, self.config.service_rate_per_replica,
        )

        return CorrectionResult(
            replicas=final,
            triggered=final > proactive_replicas,
            current_wait_estimate=wait_estimate,
        )


if __name__ == "__main__":
    config = ScalingConfig()
    corrector = ReactiveCorrector(config)

    print("Демонстрація: непередбачений сплеск, якого прогноз не побачив\n")
    scenario = [
        ("Нормальне навантаження", 120, 3),   # proactive=3 реплік вистачає
        ("Раптовий сплеск (flash-crowd)", 280, 3),  # proactive все ще 3 (прогноз не встиг)
        ("Сплеск триває", 280, 5),             # проактивний рівень вже підняв до 5 на наступному циклі
    ]

    for label, rps, proactive in scenario:
        result = corrector.correct(rps, proactive)
        status = "!! КОРЕКТОР СПРАЦЮВАВ !!" if result.triggered else "без втручання"
        print(f"{label}: RPS={rps}, proactive={proactive} -> final={result.replicas} "
              f"(очік. час у черзі при proactive: {result.current_wait_estimate:.2f}с) [{status}]")