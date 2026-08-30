"""LangSmith distributed tracing helpers.

Propagates the LangSmith trace context (the 'langsmith-trace' / 'baggage' headers)
across the HTTP boundary between microservices, so a single patient case shows up
as one connected trace tree in LangSmith instead of disconnected per-service traces.
"""
from langsmith.run_helpers import get_current_run_tree, tracing_context


class LangSmithTracingMiddleware:
    """ASGI middleware: if the incoming request carries a 'langsmith-trace' header
    (set by another service via trace_headers()), continue that trace instead of
    starting a new one. Every run created while handling the request is tagged with
    service_name so it can be filtered within the shared LangSmith project.
    """

    def __init__(self, app, service_name: str):
        self.app = app
        self.service_name = service_name

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or "headers" not in scope:
            await self.app(scope, receive, send)
            return

        headers = dict(scope["headers"])
        parent = headers if b"langsmith-trace" in headers else None
        with tracing_context(parent=parent, tags=[self.service_name]):
            await self.app(scope, receive, send)


def trace_headers() -> dict:
    """Return LangSmith distributed-tracing headers for the currently active run
    (if any), to attach to outgoing HTTP calls made to sibling services."""
    run_tree = get_current_run_tree()
    return run_tree.to_headers() if run_tree else {}
