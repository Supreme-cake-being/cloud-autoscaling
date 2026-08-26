"""
Ручна перевірка коректності розрахунків Модуля 3 на числах, які легко
перерахувати вручну — корисно для захисту: показати комісії, що логіка
не "чорна скринька", а перевіряється на простих прикладах.
"""

from config import ScalingConfig
from decision_engine import ProactiveDecisionEngine
from queuing_corrector import min_servers_for_sla, expected_wait_time


def test_decision_engine_basic_math():
    """Перевірка формули без cooldown-ускладнень (перший виклик)."""
    config = ScalingConfig(
        min_replicas=1, max_replicas=20,
        capacity_per_replica_rps=50.0, safety_factor=1.0,  # safety_factor=1 для простого рахунку
    )
    engine = ProactiveDecisionEngine(config)

    # 100 RPS / 50 RPS-на-репліку = точно 2.0 -> ceil(2.0) = 2
    decision = engine.decide(predicted_load_rps=100)
    assert decision.replicas == 2, f"Очікував 2, отримав {decision.replicas}"

    print(f"[OK] 100 RPS / 50 RPS-на-репліку -> {decision.replicas} репліки (очікував 2)")


def test_decision_engine_respects_min_max():
    """Дуже мале/велике навантаження має впертись у межі min/max."""
    config = ScalingConfig(min_replicas=2, max_replicas=10, capacity_per_replica_rps=50.0, safety_factor=1.0)
    engine = ProactiveDecisionEngine(config)

    tiny = engine.decide(predicted_load_rps=1)
    assert tiny.replicas == config.min_replicas, f"Очікував min_replicas={config.min_replicas}"
    print(f"[OK] Дуже мале навантаження (1 RPS) -> {tiny.replicas} (межа min_replicas)")

    engine2 = ProactiveDecisionEngine(config)  # новий екземпляр, без cooldown від попереднього тесту
    huge = engine2.decide(predicted_load_rps=10000)
    assert huge.replicas == config.max_replicas, f"Очікував max_replicas={config.max_replicas}"
    print(f"[OK] Величезне навантаження (10000 RPS) -> {huge.replicas} (межа max_replicas)")


def test_queuing_stability_boundary():
    """Система на межі стабільності (arrival ~ service_rate*c) має вимагати більше серверів."""
    # 100 RPS, кожен сервер тягне 50 RPS -> мінімум для стабільності: 2 сервери,
    # але при рівно 2 система працює на межі (rho->1), тому реально треба більше
    servers = min_servers_for_sla(arrival_rate=100, service_rate=50, max_wait=1.0)
    assert servers > 2, f"На межі стабільності має знадобитись більше 2 серверів, отримано {servers}"
    print(f"[OK] 100 RPS при 50 RPS/сервер (межа стабільності) -> {servers} серверів (SLA=1.0с)")


def test_queuing_more_servers_reduce_wait():
    """Більше серверів -> менший очікуваний час очікування (монотонність)."""
    wait_3 = expected_wait_time(c=3, arrival_rate=100, service_rate=50)
    wait_5 = expected_wait_time(c=5, arrival_rate=100, service_rate=50)
    assert wait_5 < wait_3, "Більше серверів має ЗМЕНШУВАТИ час очікування"
    print(f"[OK] Час очікування спадає з c: c=3 -> {wait_3:.4f}с, c=5 -> {wait_5:.4f}с")


if __name__ == "__main__":
    tests = [
        test_decision_engine_basic_math,
        test_decision_engine_respects_min_max,
        test_queuing_stability_boundary,
        test_queuing_more_servers_reduce_wait,
    ]

    print("Перевірка коректності розрахунків Модуля 3:\n")
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1

    print(f"\n{'Усі тести пройшли' if failed == 0 else f'{failed} тест(ів) провалено'}")