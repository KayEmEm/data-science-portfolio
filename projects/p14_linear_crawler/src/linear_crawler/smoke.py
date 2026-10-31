"""
Smoke run: crawls the synthetic series of demo_site in headless mode and checks
the result. No real site is contacted.

Usage: python -m linear_crawler.smoke [output folder]   (default artifacts/smoke)
"""

import csv
import sys
from pathlib import Path

import yaml

from linear_crawler import crawler, demo_site

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
)
EXPECTED_FILES = ["one.jpg", "one_1.jpg", "résumé.jpg"]


def write_config(folder, base_url):
    cookie_file = demo_site.write_cookie_file(folder / "cookies.txt")
    config = {
        "start_url": f"{base_url}/p1.html",
        "download_dir": str(folder / "files"),
        "cookie_file": str(cookie_file),
        "crawl_log": str(folder / "crawl.log"),
        "manifest_file": str(folder / "manifest.csv"),
        "state_file": str(folder / "crawl_state.json"),
        "debug_dir": str(folder / "debug_pages"),
        "display_mode": "headless",
        "user_agent": USER_AGENT,
        "timeout_seconds": 10,
        "min_page_delay": 0,
        "max_page_delay": 0,
        "min_firewall_sleep": 0,
        "max_firewall_sleep": 0,
        "max_download_retries": 1,
        "escape_key_stop": False,
        "file_link_pattern": demo_site.FILE_PATTERN,
        "next_page_pattern": demo_site.NEXT_PATTERN,
    }
    path = folder / "config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    return path


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    folder = Path(argv[0] if argv else "artifacts/smoke").resolve()
    folder.mkdir(parents=True, exist_ok=True)
    stale = ("files", "debug_pages", "manifest.csv", "crawl_state.json", "crawl.log")
    for old in (folder / name for name in stale):
        if old.is_dir():
            for item in old.iterdir():
                item.unlink()
        elif old.exists():
            old.unlink()

    site = demo_site.start()
    try:
        code = crawler.main([str(write_config(folder, site.base))])
    finally:
        demo_site.stop(site)

    saved = sorted(p.name for p in (folder / "files").iterdir() if not p.name.startswith("."))
    with open(folder / "manifest.csv", newline="", encoding="utf-8") as f:
        statuses = [row["status"] for row in csv.DictReader(f)]
    checks = {
        "exit code 0": code == crawler.EXIT_DONE,
        f"files {EXPECTED_FILES}": saved == EXPECTED_FILES,
        "manifest SAVE MISS SAVE SAVE": statuses == ["SAVE", "MISS", "SAVE", "SAVE"],
    }
    for name, ok in checks.items():
        print(f"smoke {'PASS' if ok else 'FAIL'}: {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
