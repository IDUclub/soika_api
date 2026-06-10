"""Observability config is defined here."""

from dataclasses import dataclass


@dataclass
class PrometheusConfig:
    host: str
    port: int


@dataclass
class ObservabilityConfig:
    prometheus: PrometheusConfig | None = None
