"""Standalone Session Server that proxies through the inference router.

This decouples session/TITO logic from the Miles Router, allowing sessions
to work with the SGLang Rust Router or any other backend.  Inference
requests are proxied through the router (sglang or miles), which handles
load balancing and forwarding to worker engines.
"""

import asyncio
import logging
import time

import httpx
import orjson
import setproctitle
import uvicorn
from fastapi import FastAPI, Request
from starlette.responses import Response

from miles.rollout.session.sessions import get_worker_stats, setup_session_routes

logger = logging.getLogger(__name__)


class SessionServer:
    """Lightweight FastAPI server that manages sessions and proxies inference
    requests through the inference router (sglang or miles)."""

    def __init__(self, args, backend_url: str):
        self.args = args
        self.backend_url = backend_url
        self.app = FastAPI()

        timeout = getattr(args, "miles_router_timeout", 600.0)
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=1024),
            timeout=httpx.Timeout(timeout),
        )

        # Close the httpx connection pool when uvicorn shuts down to avoid FD leaks.
        self.app.router.on_shutdown.append(self.client.aclose)

        setup_session_routes(self.app, self, args)

    async def do_proxy(
        self,
        request: Request,
        path: str,
        body: bytes | None = None,
        headers: dict | None = None,
    ) -> dict:
        url = f"{self.backend_url}/{path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"

        if body is None:
            body = await request.body()
        if headers is None:
            headers = dict(request.headers)
        headers = {
            k: v for k, v in headers.items() if k.lower() not in ("content-length", "transfer-encoding", "host")
        }

        _t_proxy_start = time.monotonic()
        try:
            response = await self.client.request(request.method, url, content=body, headers=headers)
        except httpx.TransportError as exc:
            _elapsed_ms = (time.monotonic() - _t_proxy_start) * 1000.0
            logger.warning(
                "[session-server] proxy_transport_error method=%s path=%s url=%s elapsed_ms=%.1f "
                "error_type=%s error=%s",
                request.method,
                path,
                url,
                _elapsed_ms,
                type(exc).__name__,
                exc,
            )
            error_body = orjson.dumps({"error": f"backend transport error: {type(exc).__name__}: {exc}"})
            return {
                "request_body": body,
                "response_body": error_body,
                "status_code": 502,
                "headers": {"content-type": "application/json"},
            }
        content = await response.aread()
        return {
            "request_body": body,
            "response_body": content,
            "status_code": response.status_code,
            "headers": dict(response.headers),
        }

    def build_proxy_response(self, result: dict) -> Response:
        content = result["response_body"]
        status_code = result["status_code"]

        headers = {
            k: v
            for k, v in result["headers"].items()
            if k.lower() not in ("content-length", "transfer-encoding", "server")
        }

        content_type = headers.get("content-type")

        return Response(
            content=content,
            status_code=status_code,
            headers=headers,
            media_type=content_type,
        )


async def _stats_logger_loop(worker_port, interval_seconds: float = 300.0):
    """Per-worker observability heartbeat.

    Logs counters maintained in ``sessions._worker_stats`` plus RSS/VMS
    from ``psutil`` (hard dep — see ``requirements.txt``).

    The deltas use the last emit's ``reqs_total`` snapshot to compute
    ``reqs_since_last``. This survives counter resets caused by hot reload
    (we just report a negative delta once and move on).
    """
    import psutil

    _proc = psutil.Process()

    last_reqs_total = 0
    while True:
        try:
            stats = get_worker_stats(worker_port)
            if stats is None:
                # Routes not wired yet (very early startup) — emit a sparse log.
                logger.info(
                    "[session-server] stats worker_port=%s reqs_total=0 reqs_since_last=0 "
                    "inflight_now=0 turns_completed=0",
                    worker_port,
                )
            else:
                reqs_total = stats["reqs_total"]
                inflight_now = stats["inflight"]["count"]
                turns_completed = stats["turns_completed"]
                delta = reqs_total - last_reqs_total
                last_reqs_total = reqs_total
                mi = _proc.memory_info()
                rss_mb = mi.rss / 1024.0 / 1024.0
                vms_mb = mi.vms / 1024.0 / 1024.0
                logger.info(
                    "[session-server] stats worker_port=%s reqs_total=%d reqs_since_last=%d "
                    "inflight_now=%d turns_completed=%d rss_mb=%.0f vms_mb=%.0f",
                    worker_port,
                    reqs_total,
                    delta,
                    inflight_now,
                    turns_completed,
                    rss_mb,
                    vms_mb,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[session-server] stats logger failed")
        await asyncio.sleep(interval_seconds)


def run_session_server(args, backend_url: str):
    """Entry point to start the standalone session server as a subprocess."""
    # Visible to `pkill -9 miles`; without this the daemon inherits "python".
    setproctitle.setproctitle("miles-session-server")

    # Prefix every record in this subprocess with the pid so logs across N
    # backends are grep-distinguishable. ``force=True`` overrides any prior
    # config inherited from the parent (e.g. ray-set handlers).
    logging.basicConfig(
        format="%(asctime)s pid=%(process)d %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
        force=True,
    )

    server = SessionServer(args, backend_url)

    # Schedule the per-worker stats heartbeat once the event loop is running.
    # We wire it via FastAPI's startup event so the task lives in the same loop
    # uvicorn uses to serve requests. Stored on app.state for cancel-on-shutdown.
    worker_port = getattr(args, "session_server_port", None)

    @server.app.on_event("startup")
    async def _start_stats_logger():
        server.app.state._stats_task = asyncio.create_task(_stats_logger_loop(worker_port))

    @server.app.on_event("shutdown")
    async def _stop_stats_logger():
        task = getattr(server.app.state, "_stats_task", None)
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    logger.info(
        "[session-server] Starting on %s:%s, proxying to %s",
        args.session_server_ip,
        args.session_server_port,
        backend_url,
    )
    uvicorn.run(server.app, host=args.session_server_ip, port=args.session_server_port, log_level="info")
