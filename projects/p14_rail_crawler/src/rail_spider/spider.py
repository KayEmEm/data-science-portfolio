"""
Rail crawler for a linear page series.

Each page in the series holds a link to a file and a link to the next page.
The script saves the file of each page and follows the next page link. The
series ends at the page whose next page link points to itself.

Stopping a run:
    Escape key  stops at the next safe point: after the current page load or
                download, or at once during a wait. State, manifest and
                summary are written, so --resume continues from that page.
                Needs the terminal that started the script, in the foreground.
    Ctrl+C      stops at once.

Usage:
    python -m rail_spider <cobweb.yaml> [--start-url URL]
                             [--resume | --catch-up | --fill-gaps | --gaps]
                             [--max-pages N] [--dry-run]

Positions and expiring URLs:
    The page reached from the entry URL (start_url or --start-url) is position
    1. register_file keeps one row per position, because page and file URLs can
    expire between visits. --catch-up and --fill-gaps walk from the entry URL:
    on positions already done they only read the page source for the next link
    (no browser visit, no file request), and they process in full only the
    positions recorded as failed, without a file link, or with the file missing
    from disk. --resume first tries the URL in state_file and, when that URL no
    longer gives a page with links, walks from the entry URL as --catch-up does.

Exit codes:
    0    series complete (final page points to itself), or --fill-gaps done
    1    run halted before the end of the series
    2    usage or configuration error
    3    another run holds the lock on the download folder
    4    session expired (session_expired_pattern matched, or a page
         redirected to another path while halt_on_redirect is true)
    130  manual interruption (Ctrl+C, or the Escape key in the terminal)
    143  termination signal (SIGTERM)

display_mode in the config selects how the browser window is shown:
    visible   normal browser window on the desktop
    headless  no window; Chromium's newer headless mode (channel "chromium")
    virtual   normal browser window on a private Xvfb display (needs Xvfb)

Required config values include user_agent: the "User agent" line from
chrome://version in the Chrome browser that exported the cookie file.
"""

import argparse
import contextlib
import csv
import fcntl
import hashlib
import http.client
import http.cookiejar
import json
import logging
import os
import platform
import random
import re
import select
import shutil
import signal
import subprocess
import sys
import termios
import threading
import time
import tty
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from email.message import Message
from email.utils import parsedate_to_datetime
from html import unescape
from logging.handlers import RotatingFileHandler
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

import yaml
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

# ==============================================================================
# CONSTANTS
# ==============================================================================
EXIT_DONE = 0
EXIT_HALTED = 1
EXIT_USAGE = 2
EXIT_LOCKED = 3
EXIT_SESSION = 4
EXIT_SIGINT = 130
EXIT_SIGTERM = 143

REQUIRED_KEYS = (
    "start_url",
    "download_dir",
    "cookie_file",
    "crawl_log",
    "user_agent",
    "next_page_pattern",
)
# The MAIN pattern is required; file_link_pattern is its older name.
MAIN_PATTERN_KEYS = ("main_file_link_pattern", "file_link_pattern")
USER_AGENT_HINT = (
    'user_agent: copy the "User agent" line from chrome://version '
    "in the Chrome browser that exported the cookie file."
)

DEFAULTS = {
    "display_mode": None,
    "headless": False,
    "timeout_seconds": 30,
    "max_pages": 0,
    "min_page_delay": 5,
    "max_page_delay": 15,
    "max_download_retries": 5,
    "min_firewall_sleep": 30,
    "max_firewall_sleep": 90,
    "max_retry_after_seconds": 3600,
    "max_consecutive_failures": 3,
    "max_browser_restarts": 3,
    "context_restart_pages": 0,
    "block_subresources": False,
    "stream_downloads": False,
    "reject_html_files": True,
    "session_expired_pattern": None,
    "halt_on_redirect": False,
    "cookie_expiry_warn_hours": 24,
    "debug_dir": None,
    "manifest_file": None,
    "skip_saved": True,
    "state_file": None,
    "register_file": None,
    "side_file_link_pattern": None,
    "file_number_pattern": None,
    "max_link_refreshes": 1,
    "min_walk_delay": None,
    "max_walk_delay": None,
    "console_level": "INFO",
    "log_max_bytes": 10 * 1024 * 1024,
    "log_backups": 5,
    "escape_key_stop": True,
}

BLOCK_STATUSES = frozenset({403, 429, 503})
# File responses with these statuses are not a file payload: the link is
# treated as expired and the page is loaded again for a new link. 429 and 5xx
# are retried with back off instead.
EXPIRED_STATUSES = frozenset({400, 401, 403, 404, 410})
BLOCKED_RESOURCE_TYPES = frozenset({"image", "media", "font", "stylesheet"})
HTML_CONTENT_TYPES = frozenset({"text/html", "application/xhtml+xml"})
HTTPONLY_PREFIX = "#HttpOnly_"
LOCK_NAME = ".spider.lock"
PART_NAME = ".incoming.part"
CHUNK_SIZE = 1024 * 1024
ESCAPE = b"\x1b"
# Arrow and function keys send sequences that start with Escape. A lone Escape
# is a keypress only if no further byte follows within this time.
ESCAPE_SEQUENCE_SECONDS = 0.05

DISPLAY_VISIBLE = "visible"
DISPLAY_HEADLESS = "headless"
DISPLAY_VIRTUAL = "virtual"
DISPLAY_MODES = (DISPLAY_VISIBLE, DISPLAY_HEADLESS, DISPLAY_VIRTUAL)
XVFB_SCREEN = "1920x1080x24"
XVFB_START_SECONDS = 10

# Log level names. All names have equal length.
DBUG = logging.DEBUG
INFO = logging.INFO
DONE = 25
WARN = logging.WARNING
STOP = 35
FAIL = logging.ERROR
LEVEL_NAMES = {DBUG: "DBUG", INFO: "INFO", DONE: "DONE", WARN: "WARN", STOP: "STOP", FAIL: "FAIL"}
for _level, _name in LEVEL_NAMES.items():
    logging.addLevelName(_level, _name)
CONSOLE_LEVELS = {
    "DBUG": DBUG,
    "DEBUG": DBUG,
    "INFO": INFO,
    "WARN": WARN,
    "WARNING": WARN,
    "FAIL": FAIL,
    "ERROR": FAIL,
}

# Manifest status values. All values have equal length.
ROW_SAVE = "SAVE"
ROW_SKIP = "SKIP"
ROW_FAIL = "FAIL"
ROW_MISS = "MISS"
MANIFEST_FIELDS = [
    "timestamp",
    "page_number",
    "page_url",
    "file_url",
    "saved_name",
    "size",
    "sha256",
    "status",
    "stream",
]
# Register rows use ROW_SAVE, ROW_MISS and ROW_FAIL as status. saved_name is
# relative to download_dir and includes the stream folder (MAIN/1.jpg).
REGISTER_FIELDS = ["position", "stream", "status", "saved_name", "file_number", "size", "sha256", "updated"]

# File streams. Each page can link to one MAIN file and one SIDE file; each
# stream has its own link pattern and its own folder under download_dir.
# All names have equal length.
STREAM_MAIN = "MAIN"
STREAM_SIDE = "SIDE"
STREAMS = (STREAM_MAIN, STREAM_SIDE)
GAP_SHOWN = 20

# Gap reasons. All reasons have equal length.
GAP_UNSEEN = "unseen"  # no row for this position and stream
GAP_FAILED = "failed"  # download failed
GAP_NOFILE = "nofile"  # page without a MAIN file link (a missing SIDE link is no gap)
GAP_NODISK = "nodisk"  # saved file no longer in download_dir

# Debug page reasons. All reasons have equal length.
WHY_NOFILE = "nofile"
WHY_NONEXT = "nonext"
WHY_NOLINK = "nolink"
WHY_EXPIRE = "expire"
WHY_FAILED = "failed"
WHY_CYCLED = "cycled"
WHY_REDRCT = "redrct"

LOG = logging.getLogger("crawler")


def log(level, text, **kwargs):
    LOG.log(level, text, **kwargs)


class BrowserLost(Exception):
    """The browser or page crashed or closed."""


class SessionExpired(Exception):
    """session_expired_pattern matched a page or file response."""


class Terminated(Exception):
    """SIGTERM received."""


class StopRequested(Exception):
    """The Escape key was pressed."""


class LinkExpired(Exception):
    """The file response is not a file payload; the file link has expired."""


class SeriesChanged(Exception):
    """A file number does not fit between the numbers of its neighbours."""


def on_sigterm(_signum, _frame):
    raise Terminated()


@dataclass
class FetchResult:
    status: int
    headers: dict
    size: int = 0
    sha256: str = ""
    html_text: str = None


# ==============================================================================
# CONFIGURATION AND COMMAND LINE
# ==============================================================================
def config_error(text):
    print(f"Config error: {text}", file=sys.stderr)
    sys.exit(EXIT_USAGE)


def compile_pattern(pattern, key, need_group=True, group_of="the URL"):
    try:
        compiled = re.compile(pattern, re.MULTILINE)
    except re.error as e:
        config_error(
            f"{key} is not a valid Python regex ({e}). Python does not support \\K; use a capture group."
        )
    if need_group and compiled.groups < 1:
        config_error(f"{key} needs one capture group around {group_of}.")
    return compiled


def load_config(path):
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        config_error(f"cannot read {path} ({e})")

    missing = [key for key in REQUIRED_KEYS if not raw.get(key)]
    if not any(raw.get(key) for key in MAIN_PATTERN_KEYS):
        missing.append(MAIN_PATTERN_KEYS[0])
    if missing:
        hint = f" {USER_AGENT_HINT}" if "user_agent" in missing else ""
        config_error(f"missing required keys: {', '.join(missing)}.{hint}")
    if not os.path.isfile(raw["cookie_file"]):
        config_error(f"cookie file not found: {raw['cookie_file']}")

    merged = dict(DEFAULTS)
    merged.update({k: v for k, v in raw.items() if v is not None})

    try:
        cfg = {
            "start_url": str(merged["start_url"]),
            "download_dir": str(merged["download_dir"]),
            "cookie_file": str(merged["cookie_file"]),
            "crawl_log": str(merged["crawl_log"]),
            "user_agent": str(merged["user_agent"]).strip(),
            "timeout_ms": int(float(merged["timeout_seconds"]) * 1000),
            "max_pages": int(merged["max_pages"]),
            "retries": int(merged["max_download_retries"]),
            "page_delay": (float(merged["min_page_delay"]), float(merged["max_page_delay"])),
            "firewall_sleep": (float(merged["min_firewall_sleep"]), float(merged["max_firewall_sleep"])),
            "max_retry_after": float(merged["max_retry_after_seconds"]),
            "max_consecutive_failures": int(merged["max_consecutive_failures"]),
            "max_browser_restarts": int(merged["max_browser_restarts"]),
            "context_restart_pages": int(merged["context_restart_pages"]),
            "block_subresources": bool(merged["block_subresources"]),
            "stream_downloads": bool(merged["stream_downloads"]),
            "reject_html_files": bool(merged["reject_html_files"]),
            "halt_on_redirect": bool(merged["halt_on_redirect"]),
            "cookie_expiry_warn_hours": float(merged["cookie_expiry_warn_hours"]),
            "debug_dir": merged["debug_dir"] or None,
            "manifest_file": merged["manifest_file"] or None,
            "skip_saved": bool(merged["skip_saved"]),
            "state_file": merged["state_file"] or None,
            "register_file": merged["register_file"] or None,
            "max_link_refreshes": int(merged["max_link_refreshes"]),
            "log_max_bytes": int(merged["log_max_bytes"]),
            "log_backups": int(merged["log_backups"]),
            "escape_key_stop": bool(merged["escape_key_stop"]),
        }
        # The walk delay falls back to the page delay when it is not set.
        walk_low = merged["min_walk_delay"]
        walk_high = merged["max_walk_delay"]
        cfg["walk_delay"] = (
            float(merged["min_page_delay"] if walk_low is None else walk_low),
            float(merged["max_page_delay"] if walk_high is None else walk_high),
        )
    except (TypeError, ValueError) as e:
        config_error(f"invalid value ({e})")

    # display_mode replaces the older headless key; headless is still read
    # when display_mode is absent.
    mode = merged["display_mode"]
    if mode is None:
        mode = DISPLAY_HEADLESS if bool(merged["headless"]) else DISPLAY_VISIBLE
    mode = str(mode).strip().lower()
    if mode not in DISPLAY_MODES:
        config_error(f"display_mode must be one of {', '.join(DISPLAY_MODES)}")
    cfg["display_mode"] = mode

    level_name = str(merged["console_level"]).upper()
    if level_name not in CONSOLE_LEVELS:
        config_error(f"console_level must be one of {', '.join(CONSOLE_LEVELS)}")
    cfg["console_level"] = CONSOLE_LEVELS[level_name]

    for key in ("page_delay", "firewall_sleep", "walk_delay"):
        low, high = cfg[key]
        if low < 0 or high < low:
            config_error(f"{key}: minimum must be >= 0 and <= maximum")
    for key in (
        "retries",
        "max_pages",
        "max_consecutive_failures",
        "max_browser_restarts",
        "context_restart_pages",
        "max_link_refreshes",
        "log_max_bytes",
        "log_backups",
    ):
        if cfg[key] < 0:
            config_error(f"{key} must be >= 0")
    if cfg["timeout_ms"] <= 0 or cfg["max_retry_after"] < 0:
        config_error("timeout_seconds must be > 0 and max_retry_after_seconds >= 0")

    main_key = next(key for key in MAIN_PATTERN_KEYS if raw.get(key))
    side_pattern = merged.get("side_file_link_pattern")
    cfg["link_res"] = {
        STREAM_MAIN: compile_pattern(merged[main_key], main_key),
        STREAM_SIDE: compile_pattern(side_pattern, "side_file_link_pattern") if side_pattern else None,
    }
    cfg["streams"] = [stream for stream in STREAMS if cfg["link_res"][stream] is not None]
    cfg["stream_dirs"] = {stream: os.path.join(cfg["download_dir"], stream) for stream in STREAMS}
    cfg["next_re"] = compile_pattern(merged["next_page_pattern"], "next_page_pattern")
    pattern = merged["session_expired_pattern"]
    cfg["session_re"] = (
        compile_pattern(pattern, "session_expired_pattern", need_group=False) if pattern else None
    )
    pattern = merged["file_number_pattern"]
    cfg["file_number_re"] = (
        compile_pattern(pattern, "file_number_pattern", group_of="the file number") if pattern else None
    )
    return cfg


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m rail_spider",
        description="Rail crawler for a linear page series. See the YAML config for settings.",
    )
    parser.add_argument("config", help="path to the YAML config file")
    parser.add_argument("--start-url", help="entry URL (position 1) instead of start_url")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="continue from the page saved in state_file; if that URL has expired, "
        "walk from the entry URL as --catch-up does (needs register_file)",
    )
    mode.add_argument(
        "--catch-up",
        action="store_true",
        help="walk from the entry URL past the positions already done, fill the gaps "
        "on the way, then continue the crawl (needs register_file)",
    )
    mode.add_argument(
        "--fill-gaps",
        action="store_true",
        help="walk from the entry URL, fill the gaps, and stop after the last gap (needs register_file)",
    )
    mode.add_argument(
        "--gaps",
        action="store_true",
        help="print the gaps in register_file and in download_dir, then exit; no network access",
    )
    parser.add_argument("--max-pages", type=int, help="override max_pages (0 means no limit)")
    parser.add_argument("--dry-run", action="store_true", help="log the extracted links without downloading")
    args = parser.parse_args(argv)
    if args.max_pages is not None and args.max_pages < 0:
        parser.error("--max-pages must be >= 0")
    return args


# ==============================================================================
# LOGGING, LOCK, STATE
# ==============================================================================
def setup_logging(cfg):
    LOG.setLevel(DBUG)
    LOG.handlers.clear()
    LOG.propagate = False
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(cfg["console_level"])
    console.setFormatter(formatter)
    LOG.addHandler(console)
    file_handler = RotatingFileHandler(
        cfg["crawl_log"], maxBytes=cfg["log_max_bytes"], backupCount=cfg["log_backups"], encoding="utf-8"
    )
    file_handler.setLevel(DBUG)
    file_handler.setFormatter(formatter)
    LOG.addHandler(file_handler)


def acquire_lock(download_dir):
    """Holds an exclusive lock on the download folder for the whole run."""
    # The handle stays open for the whole run: closing it releases the lock.
    handle = open(os.path.join(download_dir, LOCK_NAME), "a+", encoding="utf-8")  # noqa: SIM115
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def write_json_atomic(path, data):
    partial = path + ".part"
    with open(partial, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(partial, path)


def load_state(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def ensure_parent_dir(path):
    if path:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)


# ==============================================================================
# VIRTUAL DISPLAY
# ==============================================================================
def start_xvfb():
    """Starts Xvfb on a free display number. Returns (process, display)."""
    if shutil.which("Xvfb") is None:
        raise RuntimeError(
            "display_mode virtual needs Xvfb; install the xvfb package "
            "(Debian or Ubuntu: sudo apt install xvfb)"
        )
    read_fd, write_fd = os.pipe()
    process = subprocess.Popen(
        ["Xvfb", "-displayfd", str(write_fd), "-screen", "0", XVFB_SCREEN, "-nolisten", "tcp"],
        pass_fds=(write_fd,),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.close(write_fd)
    with os.fdopen(read_fd, "r") as reader:
        ready, _, _ = select.select([reader], [], [], XVFB_START_SECONDS)
        number = reader.readline().strip() if ready else ""
    if not number.isdigit():
        stop_process(process)
        raise RuntimeError("Xvfb did not report a display number")
    return process, f":{number}"


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def check_display(mode):
    if mode == DISPLAY_VISIBLE and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        log(
            WARN,
            "display_mode is visible, but no DISPLAY or WAYLAND_DISPLAY is set; "
            "the browser may fail to start. Consider display_mode virtual.",
        )


# ==============================================================================
# COOKIES AND USER AGENT
# ==============================================================================
def load_netscape_cookies(path):
    """
    Parses a Netscape cookie file into Playwright cookie dictionaries.
    Columns: domain, include_subdomains, path, secure, expiry, name, value.
    Lines prefixed with #HttpOnly_ are HttpOnly cookies, not comments.
    """
    cookies = []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\r\n")
            http_only = False
            if line.startswith(HTTPONLY_PREFIX):
                http_only = True
                line = line[len(HTTPONLY_PREFIX) :]
            elif line.startswith("#") or not line.strip():
                continue

            parts = line.split("\t")
            if len(parts) != 7:
                skipped += 1
                continue
            domain, _subdomains, cookie_path, secure, expiry, name, value = parts
            try:
                expires = int(float(expiry))
            except ValueError:
                skipped += 1
                continue

            cookies.append(
                {
                    "name": name,
                    "value": value,
                    "domain": domain,
                    "path": cookie_path or "/",
                    # Expiry 0 marks a session cookie; Playwright uses -1 for that.
                    "expires": expires if expires > 0 else -1,
                    "httpOnly": http_only,
                    "secure": secure.upper() == "TRUE",
                }
            )
    return cookies, skipped


def check_cookie_expiry(cookies, warn_hours):
    now = time.time()
    limit = now + warn_hours * 3600
    for cookie in cookies:
        expires = cookie.get("expires", -1)
        if expires <= 0:
            continue
        when = datetime.fromtimestamp(expires).strftime("%Y-%m-%d %H:%M:%S")
        label = f"{cookie['name']} ({cookie['domain']})"
        if expires <= now:
            log(WARN, f"Cookie {label} expired at {when}.")
        elif expires <= limit:
            log(WARN, f"Cookie {label} expires at {when}, within {warn_hours:g} hours.")


def check_user_agent(user_agent):
    if not user_agent:
        return
    system = platform.system()
    token = {"Linux": "Linux", "Windows": "Windows", "Darwin": "Mac OS X"}.get(system)
    if token and token not in user_agent:
        log(
            WARN,
            f"user_agent names a different platform than this {system} host; "
            "browser properties such as navigator.platform will not match it.",
        )


def build_cookie_jar(cookies):
    """Converts Playwright cookie dictionaries into an http.cookiejar jar."""
    jar = http.cookiejar.CookieJar()
    for c in cookies:
        domain = c["domain"]
        expires = c.get("expires", -1)
        jar.set_cookie(
            http.cookiejar.Cookie(
                version=0,
                name=c["name"],
                value=c["value"],
                port=None,
                port_specified=False,
                domain=domain,
                domain_specified=domain.startswith("."),
                domain_initial_dot=domain.startswith("."),
                path=c.get("path") or "/",
                path_specified=True,
                secure=bool(c.get("secure")),
                expires=int(expires) if expires and expires > 0 else None,
                discard=not (expires and expires > 0),
                comment=None,
                comment_url=None,
                rest={"HttpOnly": None} if c.get("httpOnly") else {},
            )
        )
    return jar


# ==============================================================================
# URL, HEADER AND FILE HELPERS
# ==============================================================================
def first_line(error):
    """Playwright error messages span several lines; the first line is enough."""
    text = str(error).strip()
    return text.splitlines()[0] if text else type(error).__name__


def normalize_url(url):
    """Lowercases scheme and host and removes the fragment for comparison."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", parts.query, ""))


def redirect_changes_path(requested_url, final_url):
    """True when the redirect leads to a different path (query is ignored)."""
    return (urlsplit(requested_url).path or "/") != (urlsplit(final_url).path or "/")


def extract_urls(pattern, source, base_url):
    """Returns absolute URLs from capture group 1 of every match, in order."""
    urls = []
    for match in pattern.finditer(source):
        captured = match.group(1)
        if captured is None:
            continue
        captured = unescape(captured).strip()
        if captured:
            urls.append(urljoin(base_url, captured))
    return urls


def content_type(value):
    if not value:
        return ""
    header = Message()
    header["content-type"] = value
    return header.get_content_type()


def choose_filename(content_disposition, file_url, page_number):
    name = None
    if content_disposition:
        header = Message()
        header["content-disposition"] = content_disposition
        name = header.get_filename()
    if not name:
        name = unquote(os.path.basename(urlsplit(file_url).path))
    name = os.path.basename(name.replace("\\", "/")).strip()
    if name in ("", ".", "..", LOCK_NAME, PART_NAME):
        name = f"page_{page_number:05d}.bin"
    return name


def unique_path(directory, name):
    """Adds _1, _2 and so on before the extension if the name is taken."""
    stem, ext = os.path.splitext(name)
    candidate = os.path.join(directory, name)
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{stem}_{counter}{ext}")
        counter += 1
    return candidate


def parse_retry_after(value):
    """Returns the Retry-After delay in seconds, or None."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def backoff_delay(attempt, low, high):
    """Doubles from low on each attempt, capped at high, with random variation."""
    capped = min(low * (2 ** (attempt - 1)), high)
    return random.uniform(0.8, 1.0) * capped


def length_mismatch(headers, size):
    """Returns the declared length when it differs from size, otherwise None."""
    declared = headers.get("content-length")
    encoding = (headers.get("content-encoding") or "identity").strip().lower()
    if declared is None or encoding != "identity":
        return None
    try:
        expected = int(declared)
    except ValueError:
        return None
    return expected if expected != size else None


# ==============================================================================
# MANIFEST
# ==============================================================================
class Manifest:
    def __init__(self, path):
        self.path = path
        self.saved = {}
        # A manifest written before the stream column existed keeps its header.
        self.fields = MANIFEST_FIELDS
        if path and os.path.isfile(path) and os.path.getsize(path) > 0:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("status") == ROW_SAVE and row.get("file_url"):
                        self.saved[row["file_url"]] = row.get("saved_name", "")
                if reader.fieldnames:
                    self.fields = list(reader.fieldnames)

    def record(
        self,
        status,
        page_number,
        page_url,
        file_url="",
        saved_name="",
        size="",
        sha256="",
        stream=STREAM_MAIN,
    ):
        if status == ROW_SAVE:
            self.saved[file_url] = saved_name
        if not self.path:
            return
        write_header = not os.path.isfile(self.path) or os.path.getsize(self.path) == 0
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "page_number": page_number,
                    "page_url": page_url,
                    "file_url": file_url,
                    "saved_name": saved_name,
                    "size": size,
                    "sha256": sha256,
                    "status": status,
                    "stream": stream,
                }
            )

    def already_saved(self, file_url, download_dir):
        name = self.saved.get(file_url)
        if name and os.path.isfile(os.path.join(download_dir, name)):
            return name
        return None


# ==============================================================================
# POSITION REGISTER
# ==============================================================================
@dataclass
class WalkPlan:
    """Positions up to furthest are done unless listed in gaps."""

    furthest: int
    gaps: dict
    stop_after: int = None

    def is_done(self, position):
        return position <= self.furthest and position not in self.gaps


class Register:
    """
    One row per position and stream (position 1 = entry page), rewritten after
    each change. Page and file URLs can expire between visits, so the position
    identifies a page, not its URL. saved_name is relative to download_dir. A
    register written before streams existed has no stream column; its rows
    count as MAIN.
    """

    def __init__(self, path, number_re=None, read_only=False):
        self.path = path
        self.number_re = number_re
        self.read_only = read_only
        self.rows = {}
        if path and os.path.isfile(path):
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    stream = row.get("stream") or STREAM_MAIN
                    try:
                        self.rows[(int(row["position"]), stream)] = dict(row, stream=stream)
                    except (KeyError, TypeError, ValueError):
                        continue

    def file_number(self, name):
        """Number from capture group 1 of file_number_pattern in a file name, or None."""
        if not self.number_re or not name:
            return None
        match = self.number_re.search(os.path.basename(name))
        if not match or match.group(1) is None:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    def set(self, position, stream, status, saved_name="", size="", sha256=""):
        number = self.file_number(saved_name)
        self.rows[(position, stream)] = {
            "position": position,
            "stream": stream,
            "status": status,
            "saved_name": saved_name,
            "file_number": "" if number is None else number,
            "size": size,
            "sha256": sha256,
            "updated": datetime.now().isoformat(timespec="seconds"),
        }
        self.write()

    def write(self):
        if not self.path or self.read_only:
            return
        partial = self.path + ".part"
        with open(partial, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=REGISTER_FIELDS)
            writer.writeheader()
            for key in sorted(self.rows, key=lambda k: (k[0], STREAMS.index(k[1]) if k[1] in STREAMS else 9)):
                writer.writerow({field: self.rows[key].get(field, "") for field in REGISTER_FIELDS})
        os.replace(partial, self.path)

    def saved_name(self, position, stream, download_dir):
        """Name of the file saved for this position and stream if it is still on disk, else None."""
        row = self.rows.get((position, stream))
        if (
            row
            and row.get("status") == ROW_SAVE
            and row.get("saved_name")
            and os.path.isfile(os.path.join(download_dir, row["saved_name"]))
        ):
            return row["saved_name"]
        return None

    def furthest(self):
        return max((position for position, _stream in self.rows), default=0)

    def gaps(self, download_dir, streams=(STREAM_MAIN,)):
        """
        {position: reason} for positions up to the furthest one where a stream
        has no saved file. Reasons name the stream, for example "MAIN failed".
        A page without a SIDE link is not a gap.
        """
        seen = {position for position, _stream in self.rows}
        found = {}
        for position in range(1, self.furthest() + 1):
            if position not in seen:
                found[position] = f"{streams[0]} {GAP_UNSEEN}"
                continue
            reasons = []
            for stream in streams:
                row = self.rows.get((position, stream))
                status = row.get("status") if row else None
                if row is None:
                    reasons.append(f"{stream} {GAP_UNSEEN}")
                elif status == ROW_FAIL:
                    reasons.append(f"{stream} {GAP_FAILED}")
                elif status == ROW_MISS:
                    if stream == STREAM_MAIN:
                        reasons.append(f"{stream} {GAP_NOFILE}")
                elif self.saved_name(position, stream, download_dir) is None:
                    reasons.append(f"{stream} {GAP_NODISK}")
            if reasons:
                found[position] = ", ".join(reasons)
        return found

    def plan(self, download_dir, streams=(STREAM_MAIN,), stop_at_last_gap=False):
        gaps = self.gaps(download_dir, streams)
        stop_after = max(gaps) if stop_at_last_gap and gaps else None
        return WalkPlan(self.furthest(), gaps, stop_after)

    def numbers(self, stream):
        """[(position, file number)] of saved positions of one stream with a number, by position."""
        pairs = []
        for (position, row_stream), row in sorted(self.rows.items()):
            value = str(row.get("file_number", ""))
            if row_stream == stream and row.get("status") == ROW_SAVE and value.isdigit():
                pairs.append((position, int(value)))
        return pairs

    def neighbour_numbers(self, position, stream):
        """File numbers of the nearest saved positions of the stream before and after, or None."""
        pairs = self.numbers(stream)
        before = [n for p, n in pairs if p < position]
        after = [n for p, n in pairs if p > position]
        return (before[-1] if before else None), (after[0] if after else None)

    def number_report(self, stream):
        """
        Compares the file numbers of neighbouring saved positions of one
        stream. Returns (skipped, open, disorder): numbers missing between
        adjacent positions, which the site skipped, since no page lies between
        them; numbers missing across other positions, which those positions
        may hold; and positions whose number is not above the number before them.
        """
        pairs = self.numbers(stream)
        skipped, open_numbers, disorder = [], [], []
        for (p1, n1), (p2, n2) in zip(pairs, pairs[1:], strict=False):
            if n2 <= n1:
                disorder.append(p2)
                continue
            missing = list(range(n1 + 1, n2))
            (skipped if p2 == p1 + 1 else open_numbers).extend(missing)
        return skipped, open_numbers, disorder


# ==============================================================================
# ESCAPE KEY
# ==============================================================================
class EscapeWatcher:
    """
    Watches the terminal for a lone Escape keypress in a background thread and
    sets stop_event. The terminal is switched to cbreak mode (no echo, keys read
    one at a time; Ctrl+C still works) and restored when the watcher stops.
    """

    def __init__(self, stop_event):
        self.stop_event = stop_event
        self.fd = None
        self.saved = None
        self.thread = None
        self.running = threading.Event()

    def foreground(self):
        try:
            return os.tcgetpgrp(self.fd) == os.getpgrp()
        except OSError:
            return False

    def start(self):
        """Returns a reason string when the watcher cannot run, otherwise None."""
        if not sys.stdin or not sys.stdin.isatty():
            return "standard input is not a terminal"
        self.fd = sys.stdin.fileno()
        if not self.foreground():
            return "the process is not in the foreground of its terminal"
        try:
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        except termios.error as e:
            self.saved = None
            return f"terminal settings cannot be changed ({e})"
        self.running.set()
        self.thread = threading.Thread(target=self.watch, name="escape-watcher", daemon=True)
        self.thread.start()
        return None

    def watch(self):
        while self.running.is_set():
            ready, _, _ = select.select([self.fd], [], [], 0.2)
            # Reading from the terminal while in the background would stop the
            # whole process (SIGTTIN), so input is read only in the foreground.
            if not ready or not self.foreground():
                if ready:
                    time.sleep(0.2)
                continue
            try:
                data = os.read(self.fd, 1)
            except OSError:
                return
            if not data:
                return
            if data != ESCAPE:
                continue
            more, _, _ = select.select([self.fd], [], [], ESCAPE_SEQUENCE_SECONDS)
            if more:
                os.read(self.fd, 64)
                continue
            if not self.stop_event.is_set():
                self.stop_event.set()
                log(STOP, "Escape key pressed. Stopping at the next safe point.")

    def stop(self):
        self.running.clear()
        if self.thread is not None:
            self.thread.join(timeout=1)
        if self.saved is not None:
            # Restoring the terminal from the background would raise SIGTTOU.
            previous = signal.signal(signal.SIGTTOU, signal.SIG_IGN)
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
            except termios.error:
                pass
            finally:
                signal.signal(signal.SIGTTOU, previous)
            self.saved = None


# ==============================================================================
# CRAWLER
# ==============================================================================
class Spider:
    def __init__(self, cfg, cookies, dry_run=False):
        self.cfg = cfg
        self.cookies = cookies
        self.dry_run = dry_run
        self.manifest = Manifest(None if dry_run else cfg["manifest_file"])
        self.register = Register(cfg["register_file"], cfg["file_number_re"], read_only=dry_run)
        # plan: positions to walk past (WalkPlan) or None for a normal crawl.
        # fallback: (entry URL, WalkPlan) used when a resumed URL has expired.
        self.plan = None
        self.fallback = None
        self.part_path = os.path.join(cfg["download_dir"], PART_NAME)
        self.stats = {"pages": 0, "walked": 0, "saved": 0, "skipped": 0, "refreshed": 0}
        self.no_file_link = []
        self.failed = []
        self.consecutive_failures = 0
        self.browser_restarts = 0
        self.pages_in_context = 0
        self.pw = None
        self.browser = None
        self.context = None
        self.page = None
        self.page_crashed = False
        self.effective_ua = None
        self.stop_event = threading.Event()
        self.xvfb = None
        self.xvfb_display = None

    # --- Browser lifecycle ---------------------------------------------------
    def launch_options(self):
        mode = self.cfg["display_mode"]
        if mode == DISPLAY_HEADLESS:
            return {"headless": True, "channel": "chromium"}
        if mode == DISPLAY_VIRTUAL:
            if self.xvfb is None or self.xvfb.poll() is not None:
                self.xvfb, self.xvfb_display = start_xvfb()
                log(INFO, f"Virtual display started: Xvfb {self.xvfb_display}")
            return {"headless": False, "env": dict(os.environ, DISPLAY=self.xvfb_display)}
        return {"headless": False}

    def start_browser(self):
        self.browser = self.pw.chromium.launch(**self.launch_options())
        self.new_context()

    def new_context(self):
        args = {"viewport": {"width": 1280, "height": 720}}
        if self.cfg["user_agent"]:
            args["user_agent"] = self.cfg["user_agent"]
        self.context = self.browser.new_context(**args)
        if self.cookies:
            self.context.add_cookies(self.cookies)
        if self.cfg["block_subresources"]:
            self.context.route("**/*", self.route_request)
        self.page = self.context.new_page()
        self.page_crashed = False
        self.page.on("crash", self.on_crash)
        self.effective_ua = self.page.evaluate("() => navigator.userAgent")
        self.pages_in_context = 0

    def close_browser(self):
        # After Ctrl+C in a terminal, the Playwright driver process has also
        # received SIGINT and exited, so close() fails with a driver error.
        for target in (self.context, self.browser):
            if target is not None:
                try:
                    target.close()
                except Exception as e:
                    log(DBUG, f"Browser close: {type(e).__name__}: {first_line(e)}")
        self.context = self.browser = self.page = None

    def on_crash(self, _page):
        self.page_crashed = True

    @staticmethod
    def route_request(route):
        if route.request.resource_type in BLOCKED_RESOURCE_TYPES:
            route.abort()
        else:
            route.continue_()

    def check_browser(self, error=None):
        """Raises BrowserLost if the browser or page is no longer usable."""
        lost = (
            self.page_crashed
            or self.page is None
            or self.page.is_closed()
            or not self.browser.is_connected()
            or (error is not None and type(error).__name__ == "TargetClosedError")
        )
        if lost:
            raise BrowserLost(first_line(error) if error else "page crashed")

    def snapshot_cookies(self):
        """Keeps current cookies so a restart does not lose cookies set by the site."""
        with contextlib.suppress(PlaywrightError):
            self.cookies = self.context.cookies()

    def maybe_restart_context(self):
        self.pages_in_context += 1
        limit = self.cfg["context_restart_pages"]
        if limit and self.pages_in_context >= limit:
            self.snapshot_cookies()
            self.context.close()
            self.new_context()
            log(INFO, f"Browser context restarted after {limit} pages.")

    # --- Waiting -------------------------------------------------------------
    def check_stop(self):
        if self.stop_event.is_set():
            raise StopRequested()

    def sleep(self, seconds, reason):
        """Waits, but returns early with StopRequested when Escape is pressed."""
        self.check_stop()
        log(INFO, f"{reason}: sleeping {seconds:.2f} seconds")
        if self.stop_event.wait(seconds):
            raise StopRequested()

    def wait_retry(self, attempt, retry_after):
        if retry_after is not None:
            delay = min(retry_after, self.cfg["max_retry_after"])
            self.sleep(
                delay, f"Retry-After {retry_after:.0f} s (capped)" if retry_after > delay else "Retry-After"
            )
        else:
            delay = backoff_delay(attempt, *self.cfg["firewall_sleep"])
            self.sleep(delay, f"Back off (attempt {attempt})")

    # --- Records -------------------------------------------------------------
    def save_state(self, url, page_number, complete=False):
        path = self.cfg["state_file"]
        if not path or self.dry_run:
            return
        # During a walk, positions up to the furthest one do not move the saved
        # position back; the end of the series is always recorded.
        if self.plan is not None and page_number <= self.plan.furthest and not complete:
            return
        write_json_atomic(
            path,
            {
                "next_url": url,
                "page_number": page_number,
                "complete": complete,
                "updated": datetime.now().isoformat(timespec="seconds"),
            },
        )

    def save_debug_page(self, number, url, source, reason):
        directory = self.cfg["debug_dir"]
        if not directory or source is None:
            return
        path = unique_path(directory, f"page_{number:05d}_{reason}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(source)
        log(INFO, f"[Page {number}] Saved source of {url} to {path}")

    def remove_part(self):
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.part_path)

    # --- Network steps -------------------------------------------------------
    def load_page(self, url):
        """
        Returns (final_url, raw_html) or None after all attempts fail.

        The browser visits the page so that page scripts run and cookies stay
        current. The raw source is then read with a direct request that uses the
        browser cookies. The source of the browser response is not used, because
        Chromium discards it when a page navigates away by itself (script
        redirect, meta refresh), and reading it can then fail or hang.
        """
        attempts = self.cfg["retries"] + 1
        for attempt in range(1, attempts + 1):
            retry_after = None
            response = None
            visited = False
            try:
                response = self.page.goto(url, wait_until="domcontentloaded", timeout=self.cfg["timeout_ms"])
                self.check_browser()
                visited = True
            except PlaywrightError as e:
                self.check_browser(e)
                if "interrupted by another navigation" in str(e):
                    log(INFO, "Page navigated by itself while loading; continuing.")
                    visited = True
                else:
                    log(WARN, f"Page load attempt {attempt}/{attempts} failed: {first_line(e)}")

            if visited:
                if response is not None and not response.ok:
                    kind = "possible block" if response.status in BLOCK_STATUSES else "error"
                    log(WARN, f"Page load attempt {attempt}/{attempts}: HTTP {response.status} ({kind})")
                    retry_after = parse_retry_after(response.headers.get("retry-after"))
                else:
                    source_url = response.url if response is not None else url
                    loaded, retry_after = self.read_source(source_url, attempt, attempts)
                    if loaded is not None:
                        final_url = loaded[0]
                        if normalize_url(self.page.url) != normalize_url(final_url):
                            log(
                                INFO,
                                f"Page navigated by itself to {self.page.url}; "
                                f"the source of {final_url} is used.",
                            )
                        return loaded

            if attempt < attempts:
                self.wait_retry(attempt, retry_after)
        return None

    def read_source(self, source_url, attempt, attempts):
        """Returns ((final_url, raw_html), None) or (None, retry_after)."""
        direct = None
        try:
            direct = self.context.request.get(source_url, timeout=self.cfg["timeout_ms"])
            if direct.ok:
                return (direct.url, direct.text()), None
            kind = "possible block" if direct.status in BLOCK_STATUSES else "error"
            log(
                WARN,
                f"Page load attempt {attempt}/{attempts}: source request "
                f"returned HTTP {direct.status} ({kind})",
            )
            return None, parse_retry_after(direct.headers.get("retry-after"))
        except PlaywrightError as e:
            self.check_browser(e)
            log(WARN, f"Page load attempt {attempt}/{attempts}: source request failed: {first_line(e)}")
            return None, None
        finally:
            if direct is not None:
                direct.dispose()

    def walk_source(self, url):
        """Page source by direct request only, no browser visit. Returns (final_url, raw_html) or None."""
        attempts = self.cfg["retries"] + 1
        for attempt in range(1, attempts + 1):
            loaded, retry_after = self.read_source(url, attempt, attempts)
            if loaded is not None:
                return loaded
            if attempt < attempts:
                self.wait_retry(attempt, retry_after)
        return None

    def fetch_with_browser(self, file_url, referer):
        response = self.context.request.get(
            file_url, headers={"Referer": referer}, timeout=self.cfg["timeout_ms"]
        )
        try:
            result = FetchResult(response.status, dict(response.headers))
            if response.ok:
                if (
                    self.cfg["reject_html_files"]
                    and content_type(result.headers.get("content-type")) in HTML_CONTENT_TYPES
                ):
                    result.html_text = response.text()
                else:
                    body = response.body()
                    with open(self.part_path, "wb") as f:
                        f.write(body)
                    result.size = len(body)
                    result.sha256 = hashlib.sha256(body).hexdigest()
            return result
        finally:
            response.dispose()

    def fetch_streaming(self, file_url, referer):
        opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(build_cookie_jar(self.cookies))
        )
        request = urllib.request.Request(
            file_url, headers={"Referer": referer, "User-Agent": self.effective_ua}
        )
        try:
            response = opener.open(request, timeout=self.cfg["timeout_ms"] / 1000)
        except urllib.error.HTTPError as e:
            headers = {k.lower(): v for k, v in e.headers.items()}
            e.close()
            return FetchResult(e.code, headers)
        with response:
            headers = {k.lower(): v for k, v in response.headers.items()}
            result = FetchResult(response.status, headers)
            if (
                self.cfg["reject_html_files"]
                and content_type(headers.get("content-type")) in HTML_CONTENT_TYPES
            ):
                result.html_text = response.read().decode("utf-8", errors="replace")
                return result
            digest = hashlib.sha256()
            with open(self.part_path, "wb") as f:
                while True:
                    self.check_stop()
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    digest.update(chunk)
                    result.size += len(chunk)
            result.sha256 = digest.hexdigest()
            return result

    def download_file(self, number, page_url, file_url, stream=STREAM_MAIN):
        """
        Saves the file into the folder of the stream. Returns (path, size,
        sha256) or None. Raises SessionExpired, or
        LinkExpired when the response is not a file payload: an HTML page, or
        an HTTP status in EXPIRED_STATUSES.
        """
        attempts = self.cfg["retries"] + 1
        for attempt in range(1, attempts + 1):
            retry_after = None
            result = None
            try:
                if self.cfg["stream_downloads"]:
                    result = self.fetch_streaming(file_url, page_url)
                else:
                    result = self.fetch_with_browser(file_url, page_url)
            except PlaywrightError as e:
                self.check_browser(e)
                log(WARN, f"Download attempt {attempt}/{attempts} failed: {first_line(e)}")
            except (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError) as e:
                log(WARN, f"Download attempt {attempt}/{attempts} failed: {type(e).__name__}: {e}")

            if result is not None:
                if result.status in EXPIRED_STATUSES:
                    self.remove_part()
                    log(
                        WARN,
                        f"[Page {number}] [{stream}] File response is HTTP {result.status}, not a file; "
                        "the link is treated as expired.",
                    )
                    raise LinkExpired(file_url)
                if not 200 <= result.status < 300:
                    kind = "possible block" if result.status in BLOCK_STATUSES else "error"
                    log(WARN, f"Download attempt {attempt}/{attempts}: HTTP {result.status} ({kind})")
                    retry_after = parse_retry_after(result.headers.get("retry-after"))
                elif result.html_text is not None:
                    self.remove_part()
                    if self.cfg["session_re"] and self.cfg["session_re"].search(result.html_text):
                        self.save_debug_page(number, file_url, result.html_text, WHY_EXPIRE)
                        raise SessionExpired(file_url)
                    log(
                        WARN,
                        f"[Page {number}] [{stream}] File response is HTML, not a file; "
                        "the link is treated as expired.",
                    )
                    self.save_debug_page(number, file_url, result.html_text, WHY_FAILED)
                    raise LinkExpired(file_url)
                else:
                    expected = length_mismatch(result.headers, result.size)
                    if expected is not None:
                        log(
                            WARN,
                            f"Download attempt {attempt}/{attempts}: received "
                            f"{result.size} bytes, Content-Length is {expected}",
                        )
                    else:
                        name = choose_filename(result.headers.get("content-disposition"), file_url, number)
                        path = unique_path(self.cfg["stream_dirs"][stream], name)
                        os.replace(self.part_path, path)
                        return path, result.size, result.sha256

            self.remove_part()
            if attempt < attempts:
                self.wait_retry(attempt, retry_after)
        return None

    # --- Page processing -----------------------------------------------------
    def check_file_number(self, number, stream, name):
        """During a walk, raises SeriesChanged if the file number does not fit its neighbours."""
        if self.plan is None or self.cfg["file_number_re"] is None:
            return
        value = self.register.file_number(name)
        if value is None:
            log(WARN, f"[Page {number}] [{stream}] file_number_pattern finds no number in {name}.")
            return
        before, after = self.register.neighbour_numbers(number, stream)
        if (before is not None and value <= before) or (after is not None and value >= after):
            raise SeriesChanged(
                f"{stream} file number {value} at position {number} is not between {before} and {after}"
            )

    def handle_file(self, number, page_url, file_url, stream=STREAM_MAIN):
        if self.dry_run:
            log(INFO, f"[Page {number}] [{stream}] Dry run. File link: {file_url}")
            return
        download_dir = self.cfg["download_dir"]
        if self.cfg["skip_saved"]:
            existing = self.register.saved_name(number, stream, download_dir)
            if existing is None:
                existing = self.manifest.already_saved(file_url, download_dir)
                if existing is not None:
                    self.register.set(number, stream, ROW_SAVE, existing)
            if existing:
                log(INFO, f"[Page {number}] [{stream}] Already saved as {existing}. Skipped.")
                self.manifest.record(ROW_SKIP, number, page_url, file_url, existing, stream=stream)
                self.stats["skipped"] += 1
                self.consecutive_failures = 0
                return

        refreshes = self.cfg["max_link_refreshes"]
        saved = None
        while True:
            log(INFO, f"[Page {number}] [{stream}] File link: {file_url}")
            try:
                saved = self.download_file(number, page_url, file_url, stream)
                break
            except LinkExpired:
                if refreshes <= 0:
                    break
                refreshes -= 1
                self.stats["refreshed"] += 1
                log(INFO, f"[Page {number}] [{stream}] Loading the page again for a new file link.")
                self.sleep(random.uniform(*self.cfg["page_delay"]), "Link refresh delay")
                file_url = self.fresh_file_link(number, page_url, stream)
                if file_url is None:
                    break

        if saved:
            path, size, sha256 = saved
            name = os.path.relpath(path, download_dir)
            log(INFO, f"[Page {number}] [{stream}] Saved {name} ({size} bytes)")
            self.manifest.record(ROW_SAVE, number, page_url, file_url, name, size, sha256, stream=stream)
            self.register.set(number, stream, ROW_SAVE, name, size, sha256)
            self.stats["saved"] += 1
            self.consecutive_failures = 0
            self.check_file_number(number, stream, name)
        else:
            log(FAIL, f"[Page {number}] [{stream}] Download failed: {file_url}. Continuing.")
            self.manifest.record(ROW_FAIL, number, page_url, file_url, stream=stream)
            self.register.set(number, stream, ROW_FAIL)
            self.failed.append(f"{stream} {file_url}")
            self.consecutive_failures += 1

    def fresh_file_link(self, number, page_url, stream=STREAM_MAIN):
        """Loads the page again and returns the file link of the stream, or None. Raises SessionExpired."""
        loaded = self.load_page(page_url)
        if loaded is None:
            log(WARN, f"[Page {number}] Page could not be loaded again; its URL may have expired.")
            return None
        _final_url, source = loaded
        session_re = self.cfg["session_re"]
        if session_re and session_re.search(source):
            self.save_debug_page(number, page_url, source, WHY_EXPIRE)
            raise SessionExpired(page_url)
        file_urls = extract_urls(self.cfg["link_res"][stream], source, page_url)
        if not file_urls:
            log(
                WARN,
                f"[Page {number}] [{stream}] Page loaded again has no file link; its URL may have expired.",
            )
            return None
        return file_urls[0]

    def follow_next(self, number, url, final_url, source, next_urls, visited):
        """Next page step shared by full pages and walk steps. Returns ("next", url) or ("end", code)."""
        if not next_urls:
            log(STOP, f"[Page {number}] Next page link absent on {final_url}. Halting.")
            self.save_debug_page(number, final_url, source, WHY_NONEXT)
            return "end", EXIT_HALTED
        if len(next_urls) > 1:
            log(WARN, f"[Page {number}] {len(next_urls)} lines match next_page_pattern; using the first.")
        next_url = next_urls[0]
        next_norm = normalize_url(next_url)

        if next_norm in (normalize_url(url), normalize_url(final_url)):
            log(DONE, f"[Page {number}] Next page link points to this page. Series complete.")
            self.save_state(final_url, number, complete=True)
            return "end", EXIT_DONE
        if next_norm in visited:
            log(
                STOP,
                f"[Page {number}] Next page link points to an earlier page "
                f"({next_url}). Cycle detected. Halting.",
            )
            self.save_debug_page(number, final_url, source, WHY_CYCLED)
            return "end", EXIT_HALTED

        self.save_state(next_url, number + 1)
        if self.cfg["max_pages"] and self.stats["pages"] >= self.cfg["max_pages"]:
            log(STOP, f"max_pages limit ({self.cfg['max_pages']}) reached. Halting.")
            return "end", EXIT_HALTED
        return "next", next_url

    def check_redirect(self, number, url, final_url, source):
        """Returns ("end", EXIT_SESSION) for a redirect to another path under halt_on_redirect, else None."""
        if normalize_url(final_url) == normalize_url(url):
            return None
        log(INFO, f"[Page {number}] Redirected to {final_url}")
        if self.cfg["halt_on_redirect"] and redirect_changes_path(url, final_url):
            log(
                STOP,
                f"[Page {number}] Redirect to another path ({final_url}); possible login redirect. Halting.",
            )
            self.save_debug_page(number, final_url, source, WHY_REDRCT)
            return "end", EXIT_SESSION
        return None

    def walk_step(self, number, url, visited):
        """
        Passes a position that is already done: reads the page source by direct
        request for the next link only. No browser visit and no file request.
        """
        loaded = self.walk_source(url)
        if loaded is None:
            log(STOP, f"[Page {number}] Walk: page source could not be read. Halting.")
            return "end", EXIT_HALTED
        final_url, source = loaded
        self.stats["walked"] += 1
        redirect = self.check_redirect(number, url, final_url, source)
        if redirect is not None:
            return redirect
        visited.add(normalize_url(url))
        visited.add(normalize_url(final_url))
        session_re = self.cfg["session_re"]
        if session_re and session_re.search(source):
            log(STOP, f"[Page {number}] session_expired_pattern matched on {final_url}. Halting.")
            self.save_debug_page(number, final_url, source, WHY_EXPIRE)
            return "end", EXIT_SESSION
        next_urls = extract_urls(self.cfg["next_re"], source, final_url)
        return self.follow_next(number, url, final_url, source, next_urls, visited)

    def process_page(self, number, url, visited, probe=False):
        """
        Returns ("next", url) to continue or ("end", exit_code) to stop. With
        probe (the first page of --resume), returns ("expired", None) when the
        page cannot be loaded, redirects to another path, or has neither link.
        """
        self.save_state(url, number)
        loaded = self.load_page(url)
        if loaded is None:
            if probe:
                return "expired", None
            log(STOP, f"[Page {number}] Page could not be loaded; neither link can be extracted. Halting.")
            return "end", EXIT_HALTED
        final_url, source = loaded
        self.stats["pages"] += 1
        self.snapshot_cookies()
        if probe and redirect_changes_path(url, final_url):
            log(INFO, f"[Page {number}] Redirected to {final_url}")
            return "expired", None
        redirect = self.check_redirect(number, url, final_url, source)
        if redirect is not None:
            return redirect
        visited.add(normalize_url(url))
        visited.add(normalize_url(final_url))

        session_re = self.cfg["session_re"]
        if session_re and session_re.search(source):
            log(STOP, f"[Page {number}] session_expired_pattern matched on {final_url}. Halting.")
            self.save_debug_page(number, final_url, source, WHY_EXPIRE)
            return "end", EXIT_SESSION

        links = {
            stream: extract_urls(self.cfg["link_res"][stream], source, final_url)
            for stream in self.cfg["streams"]
        }
        next_urls = extract_urls(self.cfg["next_re"], source, final_url)
        counts = ", ".join(f"{len(urls)} {stream}" for stream, urls in links.items())
        log(DBUG, f"[Page {number}] Matches: {counts}, {len(next_urls)} next")

        if not any(links.values()) and not next_urls:
            if probe:
                return "expired", None
            log(STOP, f"[Page {number}] File link and next page link both absent on {final_url}. Halting.")
            self.save_debug_page(number, final_url, source, WHY_NOLINK)
            return "end", EXIT_HALTED

        # --- File links, one per stream ---
        for stream, file_urls in links.items():
            if file_urls:
                if len(file_urls) > 1:
                    log(
                        WARN,
                        f"[Page {number}] [{stream}] {len(file_urls)} file link matches; using the first.",
                    )
                try:
                    self.handle_file(number, final_url, file_urls[0], stream)
                except SessionExpired:
                    log(
                        STOP,
                        f"[Page {number}] session_expired_pattern matched on the file response. Halting.",
                    )
                    return "end", EXIT_SESSION
                except SeriesChanged as e:
                    log(STOP, f"[Page {number}] The series appears changed: {e}. Halting.")
                    return "end", EXIT_HALTED
                continue
            if stream == STREAM_MAIN:
                log(
                    WARN,
                    f"[Page {number}] [{stream}] File link absent on {final_url}. No file saved. Continuing.",
                )
                self.no_file_link.append(final_url)
                self.save_debug_page(number, final_url, source, WHY_NOFILE)
            else:
                log(INFO, f"[Page {number}] [{stream}] No file link on this page.")
            if not self.dry_run:
                self.manifest.record(ROW_MISS, number, final_url, stream=stream)
                self.register.set(number, stream, ROW_MISS)

        limit = self.cfg["max_consecutive_failures"]
        if limit and self.consecutive_failures >= limit:
            log(STOP, f"{self.consecutive_failures} consecutive failed downloads. Halting.")
            return "end", EXIT_HALTED

        return self.follow_next(number, url, final_url, source, next_urls, visited)

    def loop(self, start_url, start_number, probe=False):
        visited = set()
        url = start_url
        number = start_number
        while True:
            self.check_stop()
            walking = self.plan is not None and self.plan.is_done(number)
            try:
                if walking:
                    log(INFO, f"[Page {number}] Walk past {url}")
                    kind, value = self.walk_step(number, url, visited)
                else:
                    log(INFO, f"[Page {number}] Loading {url}")
                    kind, value = self.process_page(number, url, visited, probe=probe)
                    probe = False
            except BrowserLost as e:
                self.browser_restarts += 1
                limit = self.cfg["max_browser_restarts"]
                if self.browser_restarts > limit:
                    log(STOP, f"Browser lost ({e}); restart limit ({limit}) reached. Halting.")
                    return EXIT_HALTED
                log(
                    WARN,
                    f"Browser lost ({e}). Restart {self.browser_restarts}/{limit}; "
                    "the current page is retried.",
                )
                self.close_browser()
                self.sleep(
                    backoff_delay(self.browser_restarts, *self.cfg["firewall_sleep"]), "Browser restart wait"
                )
                self.start_browser()
                continue
            if kind == "expired":
                if self.fallback is None:
                    log(
                        STOP,
                        f"[Page {number}] The saved URL no longer gives a page with links (expired). "
                        "Without register_file the crawler cannot walk from the entry URL. Halting.",
                    )
                    return EXIT_HALTED
                url, self.plan = self.fallback
                self.fallback = None
                number = 1
                visited = set()
                log(WARN, f"Saved URL has expired. Walking from the entry URL: {url}")
                log_plan(self.plan)
                continue
            if kind == "end":
                return value
            stop_after = self.plan.stop_after if self.plan is not None else None
            if not walking and stop_after is not None and number >= stop_after:
                log(DONE, f"[Page {number}] Last gap processed. Stopping.")
                return EXIT_DONE
            url = value
            number += 1
            if walking:
                self.sleep(random.uniform(*self.cfg["walk_delay"]), "Walk delay")
            else:
                self.maybe_restart_context()
                self.sleep(random.uniform(*self.cfg["page_delay"]), "Page delay")

    def summary(self):
        log(INFO, "--------------------------------------------------------")
        log(
            INFO,
            f"Pages processed: {self.stats['pages']}. Walked past: {self.stats['walked']}. "
            f"Files saved: {self.stats['saved']}. Skipped as already saved: "
            f"{self.stats['skipped']}. Link refreshes: {self.stats['refreshed']}. "
            f"Failed: {len(self.failed)}.",
        )
        for url in self.no_file_link:
            log(WARN, f"Page without file link: {url}")
        for url in self.failed:
            log(FAIL, f"Failed download: {url}")

    def run(self, start_url, start_number=1, probe=False):
        exit_code = EXIT_HALTED
        try:
            with sync_playwright() as pw:
                self.pw = pw
                try:
                    self.start_browser()
                    exit_code = self.loop(start_url, start_number, probe=probe)
                except KeyboardInterrupt:
                    log(STOP, "Manual interruption (SIGINT).")
                    exit_code = EXIT_SIGINT
                except StopRequested:
                    log(STOP, "Stopped by the Escape key. Use --resume to continue.")
                    exit_code = EXIT_SIGINT
                except Terminated:
                    log(STOP, "Termination signal (SIGTERM).")
                    exit_code = EXIT_SIGTERM
                except Exception as e:
                    log(FAIL, f"Unexpected error: {type(e).__name__}: {first_line(e)}")
                    log(DBUG, "Traceback:", exc_info=True)
                    exit_code = EXIT_HALTED
                finally:
                    self.remove_part()
                    self.close_browser()
                    stop_process(self.xvfb)
                    self.summary()
        except Exception as e:
            # Shutting down Playwright fails when its driver has already exited.
            log(DBUG, f"Playwright shutdown: {type(e).__name__}: {first_line(e)}")
        return exit_code


# ==============================================================================
# GAP REPORT
# ==============================================================================
def short_list(values):
    shown = ", ".join(str(v) for v in values[:GAP_SHOWN])
    more = len(values) - GAP_SHOWN
    return f"{shown} and {more} more" if more > 0 else shown


def log_plan(plan):
    log(INFO, f"Walk plan: furthest position {plan.furthest}, {len(plan.gaps)} gaps.")
    if plan.gaps:
        log(INFO, f"Gap positions: {short_list(sorted(plan.gaps))}")
    if plan.stop_after is not None:
        log(INFO, f"The run stops after position {plan.stop_after}.")


def report_gaps(cfg):
    """Prints the gaps recorded in register_file and found in download_dir. No network access."""
    register = Register(cfg["register_file"], cfg["file_number_re"], read_only=True)
    directory = cfg["download_dir"]
    state = load_state(cfg["state_file"]) if cfg["state_file"] else None
    complete = bool(state and state.get("complete"))
    gaps = register.gaps(directory, cfg["streams"])
    print(f"Register: {cfg['register_file']}")
    print(f"Streams: {', '.join(cfg['streams'])}")
    print(f"Furthest position: {register.furthest()}. Series complete: {'yes' if complete else 'no'}.")
    print(f"Gaps: {len(gaps)}")
    for position in sorted(gaps):
        print(f"  position {position}: {gaps[position]}")
    if cfg["file_number_re"] is not None:
        for stream in cfg["streams"]:
            skipped, open_numbers, disorder = register.number_report(stream)
            print(f"{stream} numbers skipped by the site: {short_list(skipped) or 'none'}")
            print(f"{stream} numbers missing across other positions: {short_list(open_numbers) or 'none'}")
            print(
                f"{stream} positions with a number not above the one before: {short_list(disorder) or 'none'}"
            )
    registered = {row.get("saved_name") for row in register.rows.values()}
    on_disk = []
    for folder in ("", *STREAMS):
        path = os.path.join(directory, folder)
        if not os.path.isdir(path):
            continue
        for name in os.listdir(path):
            relative = os.path.join(folder, name) if folder else name
            if name.startswith(".") or relative in registered or (not folder and name in STREAMS):
                continue
            on_disk.append(relative)
    print(f"Files in download_dir not in the register: {short_list(sorted(on_disk)) or 'none'}")
    return EXIT_DONE


# ==============================================================================
# ENTRY POINT
# ==============================================================================
def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.max_pages is not None:
        cfg["max_pages"] = args.max_pages
    walk = args.catch_up or args.fill_gaps
    if (walk or args.gaps) and not cfg["register_file"]:
        config_error("--catch-up, --fill-gaps and --gaps need register_file in the config")
    if args.gaps:
        return report_gaps(cfg)

    os.makedirs(cfg["download_dir"], exist_ok=True)
    for folder in cfg["stream_dirs"].values():
        os.makedirs(folder, exist_ok=True)
    if cfg["debug_dir"]:
        os.makedirs(cfg["debug_dir"], exist_ok=True)
    for key in ("crawl_log", "manifest_file", "state_file", "register_file"):
        ensure_parent_dir(cfg[key])
    setup_logging(cfg)

    lock = acquire_lock(cfg["download_dir"])
    if lock is None:
        log(FAIL, f"Another run holds the lock on {cfg['download_dir']}. Exiting.")
        return EXIT_LOCKED
    signal.signal(signal.SIGTERM, on_sigterm)

    try:
        log(INFO, f"--- Session start. Config: {args.config}{'. Dry run' if args.dry_run else ''} ---")
        entry_url = args.start_url or cfg["start_url"]
        start_url, start_number = entry_url, 1
        if args.resume:
            if not cfg["state_file"]:
                log(FAIL, "--resume needs state_file in the config.")
                return EXIT_USAGE
            state = load_state(cfg["state_file"])
            if not state or not state.get("next_url"):
                log(FAIL, f"No usable state in {cfg['state_file']}.")
                return EXIT_USAGE
            if state.get("complete"):
                log(DONE, "State file marks the series as complete. Nothing to do.")
                return EXIT_DONE
            start_url = state["next_url"]
            start_number = int(state.get("page_number", 1))
            log(INFO, f"Resuming at page {start_number}: {start_url}")

        cookies, skipped = load_netscape_cookies(cfg["cookie_file"])
        log(INFO, f"Cookies loaded: {len(cookies)}. Malformed lines skipped: {skipped}.")
        if not cookies:
            log(WARN, "No cookies loaded; the session runs without authentication.")
        check_cookie_expiry(cookies, cfg["cookie_expiry_warn_hours"])
        check_user_agent(cfg["user_agent"])
        log(INFO, f"Display mode: {cfg['display_mode']}")
        log(INFO, f"File streams: {', '.join(cfg['streams'])}, under {cfg['download_dir']}")
        check_display(cfg["display_mode"])

        spider = Spider(cfg, cookies, dry_run=args.dry_run)
        if walk:
            spider.plan = spider.register.plan(
                cfg["download_dir"], cfg["streams"], stop_at_last_gap=args.fill_gaps
            )
            if args.fill_gaps and not spider.plan.gaps:
                log(DONE, f"No gaps up to position {spider.plan.furthest}. Nothing to do.")
                return EXIT_DONE
            log(INFO, f"Walking from the entry URL: {entry_url}")
            log_plan(spider.plan)
        if args.resume and cfg["register_file"]:
            spider.fallback = (entry_url, spider.register.plan(cfg["download_dir"], cfg["streams"]))
        watcher = EscapeWatcher(spider.stop_event)
        if cfg["escape_key_stop"]:
            reason = watcher.start()
            if reason is None:
                log(INFO, "Press Escape in this terminal to stop at the next safe point.")
            else:
                log(INFO, f"Escape key stop is not active: {reason}.")
        try:
            return spider.run(start_url, start_number, probe=args.resume)
        finally:
            watcher.stop()
    finally:
        lock.close()


if __name__ == "__main__":
    sys.exit(main())
