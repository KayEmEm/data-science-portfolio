"""Start the dashboard, open it in headless Chromium and save one screenshot per tab."""

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from portfolio_common.config import parse_args

from taxi_demand import config

APP = config.PROJECT_DIR / "dashboard" / "app.py"
TABS = ["Overview", "Time patterns", "Zones", "Weather", "Data quality"]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for(port: int, timeout_s: float = 60) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.5)
    raise TimeoutError(f"the dashboard did not open port {port} within {timeout_s} s")


def capture(marts: Path, out_dir: Path, width: int, height: int) -> list[Path]:
    from playwright.sync_api import sync_playwright

    out_dir.mkdir(parents=True, exist_ok=True)
    port = free_port()
    command = [
        sys.executable, "-m", "streamlit", "run", str(APP),
        "--server.headless", "true", "--server.port", str(port), "--browser.gatherUsageStats", "false",
        "--theme.base", "light", "--theme.primaryColor", "#2a78d6", "--client.toolbarMode", "viewer",
        "--", "--marts", str(marts),
    ]  # fmt: skip
    server = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    saved = []
    try:
        wait_for(port)
        with sync_playwright() as pw:
            executable = os.environ.get("CHROMIUM_PATH")
            browser = pw.chromium.launch(executable_path=executable) if executable else pw.chromium.launch()
            page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
            page.goto(f"http://127.0.0.1:{port}", wait_until="networkidle")
            page.get_by_role("tab", name=TABS[0]).wait_for(timeout=60_000)
            for tab in TABS:
                page.get_by_role("tab", name=tab).click()
                page.wait_for_timeout(2500)
                path = out_dir / f"dashboard_{tab.lower().replace(' ', '_')}.png"
                page.screenshot(path=str(path), full_page=False)
                saved.append(path)
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=30)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marts", type=Path, default=config.MARTS_DIR)
    parser.add_argument("--out", type=Path, default=config.FIGURES_DIR)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=1000)
    args = parse_args(parser, __file__)
    for path in capture(args.marts, args.out, args.width, args.height):
        print(path)


if __name__ == "__main__":
    main()
