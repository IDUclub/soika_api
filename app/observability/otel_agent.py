"""Open Telemetry agent initialization is defined here."""

import platform
from functools import cache

from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import (
    SERVICE_INSTANCE_ID,
    SERVICE_NAME,
    SERVICE_VERSION,
    Resource,
)

from app.__version__ import APP_VERSION

from .config import PrometheusConfig
from .metrics_server import PrometheusServer


@cache
def get_resource() -> Resource:
    return Resource.create(
        attributes={
            SERVICE_NAME: "soika-api",
            SERVICE_VERSION: APP_VERSION,
            SERVICE_INSTANCE_ID: platform.node(),
        }
    )


class OpenTelemetryAgent:  # pylint: disable=too-few-public-methods
    def __init__(
        self,
        prometheus_config: PrometheusConfig | None,
    ):
        self._resource = get_resource()
        self._prometheus: PrometheusServer | None = None

        if prometheus_config is not None:
            self._prometheus = PrometheusServer(
                port=prometheus_config.port, host=prometheus_config.host
            )

            reader = PrometheusMetricReader()
            provider = MeterProvider(resource=self._resource, metric_readers=[reader])
            metrics.set_meter_provider(provider)

    def shutdown(self) -> None:
        """Stop metrics services if they were started."""
        if self._prometheus is not None:
            self._prometheus.shutdown()
