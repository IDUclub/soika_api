"""Observability middleware is defined here."""

import time

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.common.middlewares.middleware_utils import _normalize_path
from app.observability.metrics import Metrics


class ObservabilityMiddleware(BaseHTTPMiddleware):

    def __init__(self, app: FastAPI, metrics: Metrics):
        super().__init__(app)
        self._http_metrics = metrics.http

    async def dispatch(self, request: Request, call_next):
        path = _normalize_path(request)
        method = request.method
        self._http_metrics.requests_started.add(1, {"method": method, "path": path})
        self._http_metrics.inflight_requests.add(1)
        start = time.monotonic()
        response = await call_next(request)
        duration = time.monotonic() - start
        self._http_metrics.requests_finished.add(
            1,
            {"method": method, "path": path, "status_code": response.status_code},
        )
        self._http_metrics.request_processing_duration.record(
            duration, {"method": method, "path": path}
        )
        self._http_metrics.inflight_requests.add(-1)
        return response
