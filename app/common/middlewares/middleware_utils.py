from fastapi import Request


def _normalize_path(request: Request) -> str:
    """Normalize path to avoid high-cardinality metrics."""
    route = request.scope.get("route")
    if route and hasattr(route, "path"):
        return route.path
    return request.url.path
