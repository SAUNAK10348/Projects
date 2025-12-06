from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from threading import Thread

import uvicorn


def _start_api(host: str, port: int) -> uvicorn.Server:
    config = uvicorn.Config("src.api:app", host=host, port=port, log_level="info", reload=False)
    server = uvicorn.Server(config)
    thread = Thread(target=server.run, daemon=True)
    thread.start()
    return server


def _start_dashboard(host: str, port: int, backend_url: str, api_token: str | None) -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("FUSIONNET_BACKEND_URL", backend_url)
    if api_token:
        env.setdefault("FUSIONNET_API_TOKEN", api_token)
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "src/dashboard.py",
        "--server.headless",
        "true",
        "--server.address",
        host,
        "--server.port",
        str(port),
    ]
    return subprocess.Popen(cmd, env=env)


def run(host: str = "0.0.0.0", api_port: int = 8000, dashboard_port: int = 8501) -> None:
    backend_url = f"http://{host}:{api_port}"
    api_token = os.getenv("FUSIONNET_API_TOKEN", "") or None
    print(f"Starting FastAPI backend at {backend_url} (token required: {bool(api_token)})")
    api_server = _start_api(host, api_port)
    print(f"Starting Streamlit dashboard at http://{host}:{dashboard_port}")
    dashboard = _start_dashboard(host, dashboard_port, backend_url=backend_url, api_token=api_token)
    print("Press Ctrl+C to stop both services.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping services...")
        dashboard.send_signal(signal.SIGINT)
        dashboard.wait(timeout=10)
        api_server.should_exit = True
    finally:
        if dashboard.poll() is None:
            dashboard.terminate()


if __name__ == "__main__":  # pragma: no cover
    run()
