"""
Модуль 3.3 — Інтеграція з Kubernetes API.

Архітектурне рішення — dry_run за замовчуванням:
    На момент написання цього коду немає підключеного Kubernetes-
    кластера (ні реального, ні локального Minikube/k3s/Docker Desktop
    K8s). Клас працює у двох режимах:

    - dry_run=True (за замовчуванням) — НЕ звертається до жодного
      кластера, лише друкує, яку дію виконав би. Дозволяє тестувати
      всю логіку Модуля 3 (decision_engine + queuing_corrector) без
      реальної інфраструктури — критично для розробки на етапі,
      коли K8s ще не піднятий.
    - dry_run=False — реальні виклики через офіційний Python-клієнт
      `kubernetes` (patch Deployment scale subresource). Активується,
      коли у вас буде робочий kubeconfig (Minikube/Docker Desktop K8s/
      реальний кластер).

    Це стандартна практика — відділяти "що вирішити" (decision_engine,
    queuing_corrector — чиста логіка, легко тестується) від "як
    застосувати" (k8s_client — side effects, залежить від
    інфраструктури). На захисті це гарний аргумент: логіка
    масштабування протестована незалежно від наявності кластера.
"""

from dataclasses import dataclass


@dataclass
class ScaleResult:
    success: bool
    replicas: int
    message: str


class K8sScaler:
    def __init__(self, deployment_name: str, namespace: str = "default", dry_run: bool = True):
        self.deployment_name = deployment_name
        self.namespace = namespace
        self.dry_run = dry_run
        self._apps_api = None

        if not dry_run:
            # Імпортуємо lazily — щоб пакет `kubernetes` не був
            # обов'язковою залежністю для тих, хто лише тестує логіку
            # у dry-run режимі без встановленого kubeconfig
            from kubernetes import client, config as k8s_config

            k8s_config.load_kube_config()  # читає ~/.kube/config
            self._apps_api = client.AppsV1Api()

    def get_current_replicas(self) -> int:
        """Читає поточну кількість реплік Deployment."""
        if self.dry_run:
            print(f"[DRY RUN] отримати поточні репліки {self.deployment_name} -> (симуляція: 3)")
            return 3  # заглушка для тестування без кластера

        dep = self._apps_api.read_namespaced_deployment(self.deployment_name, self.namespace)
        return dep.spec.replicas

    def scale(self, replicas: int) -> ScaleResult:
        """
        Встановлює кількість реплік через Deployment scale subresource
        (PATCH, а не PUT усього Deployment — торкається лише поля
        spec.replicas, безпечніше при паралельних змінах іншими
        контролерами).
        """
        if self.dry_run:
            msg = f"[DRY RUN] масштабував би {self.deployment_name} (ns={self.namespace}) -> {replicas} реплік"
            print(msg)
            return ScaleResult(success=True, replicas=replicas, message=msg)

        try:
            self._apps_api.patch_namespaced_deployment_scale(
                name=self.deployment_name,
                namespace=self.namespace,
                body={"spec": {"replicas": replicas}},
            )
            msg = f"Масштабовано {self.deployment_name} -> {replicas} реплік"
            return ScaleResult(success=True, replicas=replicas, message=msg)
        except Exception as e:
            msg = f"Помилка масштабування: {type(e).__name__}: {e}"
            return ScaleResult(success=False, replicas=replicas, message=msg)


if __name__ == "__main__":
    scaler = K8sScaler(deployment_name="autoscaling-demo-service", dry_run=True)

    print("Демонстрація K8sScaler у dry-run режимі (без реального кластера):\n")
    current = scaler.get_current_replicas()
    print(f"Поточні репліки: {current}\n")

    result = scaler.scale(7)
    print(f"Результат: success={result.success}, replicas={result.replicas}")