"""
Tests for linear_crawler.

Unit tests cover the helper functions. Integration tests run the crawler as a
subprocess against the synthetic site of linear_crawler.demo_site on 127.0.0.1
with headless Chromium.

Run from the project folder:  make test
"""

import csv
import fcntl
import json
import os
import pty
import re
import select
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import pytest
import yaml

from linear_crawler import crawler, demo_site
from linear_crawler.demo_site import FILE_PATTERN, NEXT_PATTERN

COMMAND = [sys.executable, "-m", "linear_crawler"]
LINUX_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
)


# ==============================================================================
# UNIT TESTS
# ==============================================================================
def test_cookie_parser_columns_httponly_and_session(tmp_path):
    path = tmp_path / "cookies.txt"
    path.write_text(
        "# Netscape HTTP Cookie File\n"
        ".example.com\tTRUE\t/\tTRUE\t0\tsess\tabc\n"
        "#HttpOnly_.example.com\tTRUE\t/app\tFALSE\t2000000000\thoc\txyz\n"
        "example.com\tFALSE\t/\tFALSE\t2000000000\tempty\t\n"
        "broken line\n",
        encoding="utf-8",
    )
    cookies, skipped = crawler.load_netscape_cookies(str(path))
    assert skipped == 1
    by_name = {c["name"]: c for c in cookies}
    assert by_name["sess"] == {
        "name": "sess",
        "value": "abc",
        "domain": ".example.com",
        "path": "/",
        "expires": -1,
        "httpOnly": False,
        "secure": True,
    }
    assert by_name["hoc"]["httpOnly"] is True
    assert by_name["hoc"]["path"] == "/app"
    assert by_name["hoc"]["expires"] == 2000000000
    assert by_name["empty"]["value"] == ""


def test_extract_file_link_unescapes_and_resolves():
    pattern = crawler.compile_pattern(FILE_PATTERN, "f")
    source = '<a class="file" href="files/a.jpg?x=1&amp;y=2" title="x">Download file</a>'
    assert crawler.extract_urls(pattern, source, "https://h.test/s/p1.html") == [
        "https://h.test/s/files/a.jpg?x=1&y=2"
    ]


def test_next_pattern_skips_links_without_rel_next():
    pattern = crawler.compile_pattern(NEXT_PATTERN, "n")
    source = '<a href="/home">Home</a> <a href="/b" class="x" rel="next">Next</a>'
    assert crawler.extract_urls(pattern, source, "https://h.test/") == ["https://h.test/b"]


def test_patterns_apply_line_by_line():
    """Patterns run with re.MULTILINE: ^ matches at the start of each line."""
    pattern = crawler.compile_pattern(r'^<a href="([^"]*)"', "n")
    assert crawler.extract_urls(pattern, 'x\n<a href="/c">', "https://h.test/") == ["https://h.test/c"]
    assert crawler.extract_urls(pattern, 'x <a href="/c">', "https://h.test/") == []


def test_choose_filename():
    cd = "attachment; filename*=UTF-8''r%C3%A9sum%C3%A9.jpg"
    assert crawler.choose_filename(cd, "https://h/x", 1) == "résumé.jpg"
    assert crawler.choose_filename(None, "https://h/a/b%20c.png?q=1", 1) == "b c.png"
    assert crawler.choose_filename('attachment; filename="../../etc/x"', "https://h/", 1) == "x"
    assert crawler.choose_filename(None, "https://h/", 7) == "page_00007.bin"


def test_unique_path(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"1")
    (tmp_path / "a_1.jpg").write_bytes(b"1")
    assert crawler.unique_path(str(tmp_path), "a.jpg") == str(tmp_path / "a_2.jpg")


def test_normalize_url():
    assert crawler.normalize_url("HTTPS://Host.Test/p.html?a=1#top") == "https://host.test/p.html?a=1"
    assert crawler.normalize_url("https://h.test") == "https://h.test/"


def test_parse_retry_after():
    assert crawler.parse_retry_after("120") == 120.0
    later = datetime.now(UTC) + timedelta(seconds=60)
    assert 50 <= crawler.parse_retry_after(format_datetime(later, usegmt=True)) <= 61
    assert crawler.parse_retry_after("garbage") is None
    assert crawler.parse_retry_after(None) is None


def test_backoff_delay_doubles_and_caps():
    for _ in range(50):
        assert 8 <= crawler.backoff_delay(1, 10, 100) <= 10
        assert 16 <= crawler.backoff_delay(2, 10, 100) <= 20
        assert 80 <= crawler.backoff_delay(9, 10, 100) <= 100


def test_length_mismatch():
    assert crawler.length_mismatch({"content-length": "10"}, 10) is None
    assert crawler.length_mismatch({"content-length": "10"}, 7) == 10
    assert crawler.length_mismatch({"content-length": "10", "content-encoding": "gzip"}, 7) is None
    assert crawler.length_mismatch({}, 7) is None


def test_content_type():
    assert crawler.content_type("text/HTML; charset=utf-8") == "text/html"
    assert crawler.content_type("image/jpeg") == "image/jpeg"
    assert crawler.content_type(None) == ""


def test_manifest_round_trip(tmp_path):
    path = str(tmp_path / "m.csv")
    (tmp_path / "a.jpg").write_bytes(b"1")
    m = crawler.Manifest(path)
    m.record(crawler.ROW_SAVE, 1, "p1", "f1", "a.jpg", 1, "h")
    m.record(crawler.ROW_SAVE, 2, "p2", "f2", "gone.jpg", 1, "h")
    m.record(crawler.ROW_MISS, 3, "p3")
    again = crawler.Manifest(path)
    assert again.already_saved("f1", str(tmp_path)) == "a.jpg"
    assert again.already_saved("f2", str(tmp_path)) is None
    with open(path, newline="", encoding="utf-8") as f:
        assert [r["status"] for r in csv.DictReader(f)] == ["SAVE", "SAVE", "MISS"]


def test_build_cookie_jar_sends_cookies():
    import urllib.request

    jar = crawler.build_cookie_jar(
        [
            {
                "name": "a",
                "value": "1",
                "domain": ".h.test",
                "path": "/",
                "expires": -1,
                "httpOnly": True,
                "secure": False,
            },
            {
                "name": "b",
                "value": "2",
                "domain": "other.test",
                "path": "/",
                "expires": -1,
                "httpOnly": False,
                "secure": False,
            },
        ]
    )
    request = urllib.request.Request("http://www.h.test/x")
    jar.add_cookie_header(request)
    assert request.get_header("Cookie") == "a=1"


# ==============================================================================
# SYNTHETIC SITE
# ==============================================================================
@pytest.fixture
def site():
    server = demo_site.start()
    yield server
    demo_site.stop(server)


def make_config(tmp_path, site, start, **overrides):
    cookies = demo_site.write_cookie_file(tmp_path / "cookies.txt")
    cfg = {
        "start_url": site.base + start,
        "download_dir": str(tmp_path / "out"),
        "cookie_file": str(cookies),
        "crawl_log": str(tmp_path / "log" / "crawl.log"),
        "manifest_file": str(tmp_path / "manifest.csv"),
        "state_file": str(tmp_path / "state.json"),
        "debug_dir": str(tmp_path / "debug"),
        "display_mode": "headless",
        "user_agent": LINUX_UA,
        "timeout_seconds": 10,
        "min_page_delay": 0,
        "max_page_delay": 0,
        "min_firewall_sleep": 0,
        "max_firewall_sleep": 0,
        "max_download_retries": 1,
        "file_link_pattern": FILE_PATTERN,
        "next_page_pattern": NEXT_PATTERN,
    }
    cfg.update(overrides)
    path = tmp_path / "config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return str(path)


def run(config, *args, timeout=120):
    return subprocess.run([*COMMAND, config, *args], capture_output=True, text=True, timeout=timeout)


def manifest_rows(tmp_path):
    with open(tmp_path / "manifest.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def out_files(tmp_path):
    return sorted(n for n in os.listdir(tmp_path / "out") if not n.startswith("."))


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================
@pytest.mark.parametrize("stream", [False, True])
def test_full_series_then_rerun_skips(tmp_path, site, stream):
    config = make_config(tmp_path, site, "/p1.html", stream_downloads=stream, block_subresources=True)
    result = run(config)
    assert result.returncode == 0, result.stdout + result.stderr
    assert out_files(tmp_path) == ["one.jpg", "one_1.jpg", "résumé.jpg"]
    assert (tmp_path / "out" / "résumé.jpg").read_bytes() == b"FILEDATA/files/cd"
    assert [r["status"] for r in manifest_rows(tmp_path)] == ["SAVE", "MISS", "SAVE", "SAVE"]
    assert json.loads((tmp_path / "state.json").read_text())["complete"] is True
    assert (tmp_path / "debug" / "page_00002_nofile.html").is_file()
    assert not any(p.endswith(".png") for p in site.requests), "subresources not blocked"

    # Rerun from start_url: every file is skipped, no new files.
    result = run(config, "--start-url", site.base + "/p1.html")
    assert result.returncode == 0
    assert out_files(tmp_path) == ["one.jpg", "one_1.jpg", "résumé.jpg"]
    statuses = [r["status"] for r in manifest_rows(tmp_path)][4:]
    assert statuses == ["SKIP", "MISS", "SKIP", "SKIP"]


def test_subresources_load_when_not_blocked(tmp_path, site):
    config = make_config(tmp_path, site, "/p4.html", block_subresources=False)
    assert run(config).returncode == 0
    assert any(p.endswith(".png") for p in site.requests)


def test_both_links_absent_halts(tmp_path, site):
    config = make_config(tmp_path, site, "/empty.html")
    result = run(config)
    assert result.returncode == 1
    assert "both absent" in result.stdout
    assert (tmp_path / "debug" / "page_00001_nolink.html").is_file()


def test_html_file_response_and_consecutive_failure_halt(tmp_path, site):
    config = make_config(tmp_path, site, "/h1.html", max_consecutive_failures=1)
    result = run(config)
    assert result.returncode == 1
    assert "File response is HTML" in result.stdout
    assert "1 consecutive failed downloads" in result.stdout
    assert [r["status"] for r in manifest_rows(tmp_path)] == ["FAIL"]
    assert (tmp_path / "debug" / "page_00001_failed.html").is_file()


@pytest.mark.parametrize("start", ["/login.html", "/s1.html"])
def test_session_expired_pattern(tmp_path, site, start):
    config = make_config(tmp_path, site, start, session_expired_pattern="Please sign in")
    result = run(config)
    assert result.returncode == 4, result.stdout
    assert (tmp_path / "debug" / "page_00001_expire.html").is_file()


def test_retry_after_is_used(tmp_path, site):
    config = make_config(tmp_path, site, "/r1.html")
    result = run(config)
    assert result.returncode == 0, result.stdout
    assert "Retry-After: sleeping 1.00 seconds" in result.stdout
    assert out_files(tmp_path) == ["limited"]


def test_content_length_mismatch_is_retried_then_fails(tmp_path, site):
    config = make_config(tmp_path, site, "/c1.html", stream_downloads=True, max_consecutive_failures=0)
    result = run(config)
    assert result.returncode == 0, result.stdout
    # A short body is reported either as a length mismatch or as IncompleteRead.
    assert "Download attempt 1/2" in result.stdout
    assert "Download attempt 2/2" in result.stdout
    assert out_files(tmp_path) == []
    assert [r["status"] for r in manifest_rows(tmp_path)] == ["FAIL"]


def test_dry_run_downloads_nothing(tmp_path, site):
    config = make_config(tmp_path, site, "/p1.html")
    result = run(config, "--dry-run")
    assert result.returncode == 0
    assert result.stdout.count("Dry run. File link") == 3
    assert out_files(tmp_path) == []
    assert not (tmp_path / "manifest.csv").exists()
    assert not (tmp_path / "state.json").exists()


def test_max_pages_then_resume(tmp_path, site):
    config = make_config(tmp_path, site, "/p1.html")
    result = run(config, "--max-pages", "2")
    assert result.returncode == 1
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["next_url"].endswith("/p3.html") and state["page_number"] == 3
    result = run(config, "--resume")
    assert result.returncode == 0, result.stdout
    assert "Resuming at page 3" in result.stdout
    assert out_files(tmp_path) == ["one.jpg", "one_1.jpg", "résumé.jpg"]
    result = run(config, "--resume")
    assert result.returncode == 0 and "Nothing to do" in result.stdout


def test_lock_blocks_second_run(tmp_path, site):
    config = make_config(tmp_path, site, "/p1.html")
    os.makedirs(tmp_path / "out")
    with open(tmp_path / "out" / ".crawler.lock", "a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert run(config).returncode == 3


def test_sigterm_writes_summary(tmp_path, site):
    config = make_config(tmp_path, site, "/p1.html", min_page_delay=30, max_page_delay=30)
    process = subprocess.Popen(
        [*COMMAND, config], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    log_path = tmp_path / "log" / "crawl.log"
    deadline = time.time() + 60
    while time.time() < deadline:
        if log_path.exists() and "Page delay" in log_path.read_text(encoding="utf-8"):
            break
        time.sleep(0.2)
    process.send_signal(signal.SIGTERM)
    output, _ = process.communicate(timeout=60)
    assert process.returncode == 143, output
    assert "Termination signal (SIGTERM)" in output
    assert "Pages processed: 1" in output


def test_context_restart_keeps_cookies(tmp_path, site):
    config = make_config(tmp_path, site, "/p1.html", context_restart_pages=1)
    result = run(config)
    assert result.returncode == 0, result.stdout
    assert result.stdout.count("Browser context restarted") == 3
    assert out_files(tmp_path) == ["one.jpg", "one_1.jpg", "résumé.jpg"]


def test_user_agent_platform_warning(tmp_path, site):
    config = make_config(tmp_path, site, "/p4.html", user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
    result = run(config)
    assert "different platform" in result.stdout


def test_cookie_expiry_warning(tmp_path, site):
    config = make_config(tmp_path, site, "/p4.html", cookie_expiry_warn_hours=10**7)
    result = run(config)
    assert "Cookie session (127.0.0.1) expires at" in result.stdout


@pytest.mark.parametrize("close_every_time", [False, True])
def test_browser_lost_restart(tmp_path, site, monkeypatch, close_every_time):
    """Closes the browser before a page load to simulate a crash."""
    config = make_config(tmp_path, site, "/p4.html", max_browser_restarts=1)
    original = crawler.Crawler.load_page
    calls = {"n": 0}

    def closing_load_page(self, url):
        calls["n"] += 1
        if close_every_time or calls["n"] == 1:
            self.browser.close()
        return original(self, url)

    monkeypatch.setattr(crawler.Crawler, "load_page", closing_load_page)
    code = crawler.main([config])
    log_text = (tmp_path / "log" / "crawl.log").read_text(encoding="utf-8")
    assert "Restart 1/1" in log_text
    if close_every_time:
        assert code == 1
        assert "restart limit (1) reached" in log_text
    else:
        assert code == 0, log_text
        assert out_files(tmp_path) == ["one.jpg"]


def test_redirect_changes_path():
    assert crawler.redirect_changes_path("https://h/a.html", "https://h/login")
    assert not crawler.redirect_changes_path("http://h/a.html?x=1", "https://www.h/a.html")
    assert not crawler.redirect_changes_path("https://h", "https://h/")


def test_redirect_to_other_path_halts(tmp_path, site):
    config = make_config(tmp_path, site, "/old.html", halt_on_redirect=True)
    result = run(config)
    assert result.returncode == 4, result.stdout
    assert "Redirect to another path" in result.stdout
    assert (tmp_path / "debug" / "page_00001_redrct.html").is_file()


def test_redirect_same_path_continues(tmp_path, site):
    config = make_config(tmp_path, site, "/p4.html?go=1", halt_on_redirect=True)
    result = run(config)
    assert result.returncode == 0, result.stdout
    assert out_files(tmp_path) == ["one.jpg"]


@pytest.mark.parametrize("value", [None, ""])
def test_user_agent_is_required(tmp_path, site, value):
    config = make_config(tmp_path, site, "/p4.html", user_agent=value)
    result = run(config)
    assert result.returncode == 2
    assert "user_agent" in result.stderr and "chrome://version" in result.stderr


def test_user_agent_is_sent(tmp_path, site):
    config = make_config(tmp_path, site, "/p4.html")
    assert run(config).returncode == 0
    assert site.user_agents and set(site.user_agents) == {LINUX_UA}


def test_display_mode_from_old_headless_key(tmp_path, site):
    config = make_config(tmp_path, site, "/p4.html", display_mode=None, headless=True)
    result = run(config)
    assert result.returncode == 0, result.stdout
    assert "Display mode: headless" in result.stdout


def test_display_mode_invalid(tmp_path, site):
    config = make_config(tmp_path, site, "/p4.html", display_mode="hidden")
    result = run(config)
    assert result.returncode == 2
    assert "display_mode must be one of" in result.stderr


@pytest.mark.skipif(not shutil.which("Xvfb"), reason="Xvfb not installed")
def test_display_mode_virtual(tmp_path, site):
    config = make_config(tmp_path, site, "/p1.html", display_mode="virtual")
    result = run(config)
    assert result.returncode == 0, result.stdout
    assert "Virtual display started: Xvfb :" in result.stdout
    assert out_files(tmp_path) == ["one.jpg", "one_1.jpg", "résumé.jpg"]
    number = re.search(r"Virtual display started: Xvfb :(\d+)", result.stdout).group(1)
    assert not os.path.exists(f"/tmp/.X{number}-lock"), "Xvfb still running after the run"


@pytest.mark.skipif(not shutil.which("xvfb-run"), reason="xvfb-run not installed")
def test_display_mode_visible(tmp_path, site):
    """Runs the visible mode on a display provided by xvfb-run."""
    config = make_config(tmp_path, site, "/p4.html", display_mode="visible")
    result = subprocess.run(["xvfb-run", "-a", *COMMAND, config], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Display mode: visible" in result.stdout
    assert "no DISPLAY" not in result.stdout


@pytest.mark.parametrize("start", ["/j1.html", "/m1.html", "/t1.html"])
def test_page_that_navigates_by_itself(tmp_path, site, start):
    """Script redirect, meta refresh or delayed redirect right after loading."""
    config = make_config(tmp_path, site, start, halt_on_redirect=True)
    result = run(config, timeout=60)
    assert result.returncode == 0, result.stdout
    assert out_files(tmp_path) == ["one.jpg"]
    assert "Series complete" in result.stdout
    assert "Redirect to another path" not in result.stdout
    assert "Protocol error" not in result.stdout
    # One browser visit plus one source request for the page.
    assert site.counts[start] == 2


def run_in_terminal(config, actions, timeout=120):
    """
    Runs the script on a pseudo terminal, followed by "stty -a" to show the
    terminal settings afterwards. actions: list of (text, count, bytes): when
    text has appeared count times in the output, bytes are typed.
    """
    command = f"{sys.executable} -m linear_crawler {config}; echo EXIT=$?; stty -a"
    pid, master = pty.fork()
    if pid == 0:
        os.execv("/bin/sh", ["/bin/sh", "-c", command])
    output = b""
    pending = list(actions)
    deadline = time.time() + timeout
    while time.time() < deadline:
        ready, _, _ = select.select([master], [], [], 0.2)
        if ready:
            try:
                chunk = os.read(master, 4096)
            except OSError:
                break
            if not chunk:
                break
            output += chunk
        while pending and output.count(pending[0][0].encode()) >= pending[0][1]:
            os.write(master, pending.pop(0)[2])
    else:
        os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    os.close(master)
    return output.decode("utf-8", errors="replace")


# pty.fork in a test process that runs a server thread; the child calls exec at once.
pytestmark_pty = pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")


@pytestmark_pty
def test_escape_key_stops_and_arrow_key_does_not(tmp_path, site):
    config = make_config(tmp_path, site, "/p1.html", min_page_delay=3, max_page_delay=3)
    output = run_in_terminal(config, [("Page delay", 1, b"\x1b[A"), ("Page delay", 2, b"\x1b")])
    assert "Press Escape in this terminal" in output
    assert "EXIT=130" in output, output
    assert "Escape key pressed" in output
    assert "Stopped by the Escape key" in output
    assert "Pages processed: 2" in output  # the arrow key did not stop the run
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["next_url"].endswith("/p3.html") and state["page_number"] == 3
    # Terminal settings restored: canonical mode and echo are on again.
    assert re.search(r"(?<![-\w])icanon", output) and re.search(r"(?<![-\w])echo(?![\w])", output)


def test_escape_key_inactive_without_terminal(tmp_path, site):
    config = make_config(tmp_path, site, "/p4.html")
    result = subprocess.run(
        [*COMMAND, config], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0
    assert "Escape key stop is not active: standard input is not a terminal" in result.stdout


@pytestmark_pty
def test_escape_key_stop_disabled(tmp_path, site):
    config = make_config(tmp_path, site, "/p4.html", escape_key_stop=False)
    output = run_in_terminal(config, [])
    assert "EXIT=0" in output
    assert "Press Escape" not in output and "not active" not in output


@pytestmark_pty
def test_ctrl_c_still_works_in_cbreak_mode(tmp_path, site):
    config = make_config(tmp_path, site, "/p1.html", min_page_delay=3, max_page_delay=3)
    output = run_in_terminal(config, [("Page delay", 1, b"\x03")])
    assert "Manual interruption (SIGINT)" in output, output
    assert "Pages processed: 1" in output
