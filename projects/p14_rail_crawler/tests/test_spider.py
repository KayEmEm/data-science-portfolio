"""
Tests for rail_crawler.

Unit tests cover the helper functions. Integration tests run the crawler as a
subprocess against the synthetic site of rail_crawler.demo_site on 127.0.0.1
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

from rail_crawler import demo_site, spider
from rail_crawler.demo_site import FILE_PATTERN, NEXT_PATTERN

COMMAND = [sys.executable, "-m", "rail_crawler"]
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
    cookies, skipped = spider.load_netscape_cookies(str(path))
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
    pattern = spider.compile_pattern(FILE_PATTERN, "f")
    source = '<a class="file" href="files/a.jpg?x=1&amp;y=2" title="x">Download file</a>'
    assert spider.extract_urls(pattern, source, "https://h.test/s/p1.html") == [
        "https://h.test/s/files/a.jpg?x=1&y=2"
    ]


def test_next_pattern_skips_links_without_rel_next():
    pattern = spider.compile_pattern(NEXT_PATTERN, "n")
    source = '<a href="/home">Home</a> <a href="/b" class="x" rel="next">Next</a>'
    assert spider.extract_urls(pattern, source, "https://h.test/") == ["https://h.test/b"]


def test_patterns_apply_line_by_line():
    """Patterns run with re.MULTILINE: ^ matches at the start of each line."""
    pattern = spider.compile_pattern(r'^<a href="([^"]*)"', "n")
    assert spider.extract_urls(pattern, 'x\n<a href="/c">', "https://h.test/") == ["https://h.test/c"]
    assert spider.extract_urls(pattern, 'x <a href="/c">', "https://h.test/") == []


def test_choose_filename():
    cd = "attachment; filename*=UTF-8''r%C3%A9sum%C3%A9.jpg"
    assert spider.choose_filename(cd, "https://h/x", 1) == "résumé.jpg"
    assert spider.choose_filename(None, "https://h/a/b%20c.png?q=1", 1) == "b c.png"
    assert spider.choose_filename('attachment; filename="../../etc/x"', "https://h/", 1) == "x"
    assert spider.choose_filename(None, "https://h/", 7) == "page_00007.bin"


def test_unique_path(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"1")
    (tmp_path / "a_1.jpg").write_bytes(b"1")
    assert spider.unique_path(str(tmp_path), "a.jpg") == str(tmp_path / "a_2.jpg")


def test_normalize_url():
    assert spider.normalize_url("HTTPS://Host.Test/p.html?a=1#top") == "https://host.test/p.html?a=1"
    assert spider.normalize_url("https://h.test") == "https://h.test/"


def test_parse_retry_after():
    assert spider.parse_retry_after("120") == 120.0
    later = datetime.now(UTC) + timedelta(seconds=60)
    assert 50 <= spider.parse_retry_after(format_datetime(later, usegmt=True)) <= 61
    assert spider.parse_retry_after("garbage") is None
    assert spider.parse_retry_after(None) is None


def test_backoff_delay_doubles_and_caps():
    for _ in range(50):
        assert 8 <= spider.backoff_delay(1, 10, 100) <= 10
        assert 16 <= spider.backoff_delay(2, 10, 100) <= 20
        assert 80 <= spider.backoff_delay(9, 10, 100) <= 100


def test_length_mismatch():
    assert spider.length_mismatch({"content-length": "10"}, 10) is None
    assert spider.length_mismatch({"content-length": "10"}, 7) == 10
    assert spider.length_mismatch({"content-length": "10", "content-encoding": "gzip"}, 7) is None
    assert spider.length_mismatch({}, 7) is None


def test_content_type():
    assert spider.content_type("text/HTML; charset=utf-8") == "text/html"
    assert spider.content_type("image/jpeg") == "image/jpeg"
    assert spider.content_type(None) == ""


def test_manifest_round_trip(tmp_path):
    path = str(tmp_path / "m.csv")
    (tmp_path / "a.jpg").write_bytes(b"1")
    m = spider.Manifest(path)
    m.record(spider.ROW_SAVE, 1, "p1", "f1", "a.jpg", 1, "h")
    m.record(spider.ROW_SAVE, 2, "p2", "f2", "gone.jpg", 1, "h")
    m.record(spider.ROW_MISS, 3, "p3")
    again = spider.Manifest(path)
    assert again.already_saved("f1", str(tmp_path)) == "a.jpg"
    assert again.already_saved("f2", str(tmp_path)) is None
    with open(path, newline="", encoding="utf-8") as f:
        assert [r["status"] for r in csv.DictReader(f)] == ["SAVE", "SAVE", "MISS"]


def test_build_cookie_jar_sends_cookies():
    import urllib.request

    jar = spider.build_cookie_jar(
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


def out_files(tmp_path, stream="MAIN"):
    return sorted(n for n in os.listdir(tmp_path / "out" / stream) if not n.startswith("."))


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================
@pytest.mark.parametrize("stream", [False, True])
def test_full_series_then_rerun_skips(tmp_path, site, stream):
    config = make_config(tmp_path, site, "/p1.html", stream_downloads=stream, block_subresources=True)
    result = run(config)
    assert result.returncode == 0, result.stdout + result.stderr
    assert out_files(tmp_path) == ["one.jpg", "one_1.jpg", "résumé.jpg"]
    assert (tmp_path / "out" / "MAIN" / "résumé.jpg").read_bytes() == b"FILEDATA/files/cd"
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
    with open(tmp_path / "out" / ".spider.lock", "a+") as handle:
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
    original = spider.Spider.load_page
    calls = {"n": 0}

    def closing_load_page(self, url):
        calls["n"] += 1
        if close_every_time or calls["n"] == 1:
            self.browser.close()
        return original(self, url)

    monkeypatch.setattr(spider.Spider, "load_page", closing_load_page)
    code = spider.main([config])
    log_text = (tmp_path / "log" / "crawl.log").read_text(encoding="utf-8")
    assert "Restart 1/1" in log_text
    if close_every_time:
        assert code == 1
        assert "restart limit (1) reached" in log_text
    else:
        assert code == 0, log_text
        assert out_files(tmp_path) == ["one.jpg"]


def test_redirect_changes_path():
    assert spider.redirect_changes_path("https://h/a.html", "https://h/login")
    assert not spider.redirect_changes_path("http://h/a.html?x=1", "https://www.h/a.html")
    assert not spider.redirect_changes_path("https://h", "https://h/")


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
    command = f"{sys.executable} -m rail_crawler {config}; echo EXIT=$?; stty -a"
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


# ==============================================================================
# EXPIRING LINKS, POSITION REGISTER, WALK AND GAPS
# ==============================================================================
TOKEN_FILES = ["1.jpg", "2.jpg", "4.jpg", "6.jpg", "7.jpg"]
NUMBER_PATTERN = r"^(\d+)\.jpg$"


def test_register_gaps_plan_and_numbers(tmp_path):
    number_re = spider.compile_pattern(NUMBER_PATTERN, "n", group_of="the file number")
    path = str(tmp_path / "register.csv")
    register = spider.Register(path, number_re)
    (tmp_path / "MAIN").mkdir()
    for name in ("1.jpg", "2.jpg", "6.jpg", "7.jpg"):
        (tmp_path / "MAIN" / name).write_bytes(b"1")
    main, side = spider.STREAM_MAIN, spider.STREAM_SIDE
    register.set(1, main, spider.ROW_SAVE, "MAIN/1.jpg")
    register.set(2, main, spider.ROW_SAVE, "MAIN/2.jpg")
    register.set(3, main, spider.ROW_MISS)
    register.set(5, main, spider.ROW_SAVE, "MAIN/6.jpg")
    register.set(6, main, spider.ROW_SAVE, "MAIN/7.jpg")
    register.set(7, main, spider.ROW_FAIL)
    register.set(8, main, spider.ROW_SAVE, "MAIN/9.jpg")  # file deleted from disk
    for position in (1, 2, 3, 5, 6, 7, 8):
        register.set(position, side, spider.ROW_MISS)
    register.set(6, side, spider.ROW_FAIL)
    again = spider.Register(path, number_re)
    assert again.gaps(str(tmp_path)) == {
        3: "MAIN nofile",
        4: "MAIN unseen",
        7: "MAIN failed",
        8: "MAIN nodisk",
    }
    # With both streams: a SIDE failure is a gap, a missing SIDE link is not.
    both = again.gaps(str(tmp_path), spider.STREAMS)
    assert both[6] == "SIDE failed" and both[7] == "MAIN failed" and 2 not in both
    plan = again.plan(str(tmp_path), stop_at_last_gap=True)
    assert plan.furthest == 8 and plan.stop_after == 8
    assert plan.is_done(1) and not plan.is_done(3) and not plan.is_done(9)
    # 3, 4 and 5 lie across gap positions 3 and 4; 8 lies across gap position 7.
    assert again.number_report(main) == ([], [3, 4, 5, 8], [])
    assert again.neighbour_numbers(4, main) == (2, 6)
    assert again.number_report(side) == ([], [], [])


def test_register_without_stream_column_reads_as_main(tmp_path):
    path = tmp_path / "register.csv"
    path.write_text(
        "position,status,saved_name,file_number,size,sha256,updated\n1,SAVE,1.jpg,1,1,h,2026-01-01T00:00:00\n",
        encoding="utf-8",
    )
    (tmp_path / "1.jpg").write_bytes(b"1")
    register = spider.Register(str(path))
    assert register.saved_name(1, spider.STREAM_MAIN, str(tmp_path)) == "1.jpg"


def test_number_report_site_skip_and_disorder(tmp_path):
    number_re = spider.compile_pattern(NUMBER_PATTERN, "n", group_of="the file number")
    register = spider.Register(None, number_re)
    main = spider.STREAM_MAIN
    register.set(1, main, spider.ROW_SAVE, "MAIN/1.jpg")
    register.set(2, main, spider.ROW_SAVE, "MAIN/3.jpg")
    register.set(3, main, spider.ROW_SAVE, "MAIN/2.jpg")
    assert register.number_report(main) == ([2], [], [3])


def test_walk_modes_need_register(tmp_path, site):
    config = make_config(tmp_path, site, "/p1.html")
    for option in ("--catch-up", "--fill-gaps", "--gaps"):
        result = run(config, option)
        assert result.returncode == 2
        assert "need register_file" in result.stderr


def token_config(tmp_path, site, **overrides):
    overrides.setdefault("register_file", str(tmp_path / "register.csv"))
    overrides.setdefault("file_number_pattern", NUMBER_PATTERN)
    config = make_config(tmp_path, site, "/e1.html", **overrides)
    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["start_url"] = demo_site.entry_url(site)
    with open(config, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return config


def counts_since(site, before):
    return {path: n - before.get(path, 0) for path, n in site.counts.items() if n != before.get(path, 0)}


def test_rerun_with_expired_links_skips_by_position(tmp_path, site):
    config = token_config(tmp_path, site)
    assert run(config).returncode == 0
    assert out_files(tmp_path) == TOKEN_FILES
    site.token = "b2"
    before = dict(site.counts)
    result = run(config, "--start-url", demo_site.entry_url(site))
    assert result.returncode == 0, result.stdout
    assert out_files(tmp_path) == TOKEN_FILES  # no second copy with a suffix
    statuses = [r["status"] for r in manifest_rows(tmp_path)][6:]
    assert statuses == ["SKIP", "SKIP", "MISS", "SKIP", "SKIP", "SKIP"]
    assert not any(path.startswith("/files/") for path in counts_since(site, before))


def test_expired_file_link_reloads_page(tmp_path, site):
    site.expire_once.add(2)
    config = token_config(tmp_path, site)
    result = run(config)
    assert result.returncode == 0, result.stdout
    assert "File response is HTML, not a file; the link is treated as expired." in result.stdout
    assert "Loading the page again for a new file link" in result.stdout
    assert "Link refreshes: 1" in result.stdout
    assert out_files(tmp_path) == TOKEN_FILES
    assert site.counts["/e2.html"] == 4  # two loads, each a browser visit and a source request


def test_expired_file_link_fails_without_refresh(tmp_path, site):
    site.expire_once.add(2)
    config = token_config(tmp_path, site, max_link_refreshes=0, max_consecutive_failures=0)
    result = run(config)
    assert result.returncode == 0, result.stdout
    assert "2.jpg" not in out_files(tmp_path)
    assert [r["status"] for r in manifest_rows(tmp_path)][1] == "FAIL"


def test_resume_after_expiry_walks_from_entry(tmp_path, site):
    config = token_config(tmp_path, site)
    assert run(config, "--max-pages", "2").returncode == 1
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["page_number"] == 3 and "/e3.html?t=a1" in state["next_url"]
    site.token = "b2"
    before = dict(site.counts)
    result = run(config, "--resume", "--start-url", demo_site.entry_url(site))
    assert result.returncode == 0, result.stdout
    assert "Saved URL has expired. Walking from the entry URL" in result.stdout
    assert "[Page 1] Walk past" in result.stdout and "[Page 2] Walk past" in result.stdout
    assert out_files(tmp_path) == TOKEN_FILES
    since = counts_since(site, before)
    assert since["/e1.html"] == 1 and since["/e2.html"] == 1  # source request only
    assert "/files/n1" not in since and "/files/n2" not in since


def test_resume_after_expiry_without_register_halts(tmp_path, site):
    config = token_config(tmp_path, site, register_file=None)
    assert run(config, "--max-pages", "2").returncode == 1
    site.token = "b2"
    result = run(config, "--resume")
    assert result.returncode == 1
    assert "Without register_file the crawler cannot walk" in result.stdout


def test_catch_up_continues_after_furthest(tmp_path, site):
    config = token_config(tmp_path, site)
    assert run(config, "--max-pages", "3").returncode == 1
    site.token = "b2"
    before = dict(site.counts)
    result = run(config, "--catch-up", "--start-url", demo_site.entry_url(site))
    assert result.returncode == 0, result.stdout
    assert "Walk plan: furthest position 3, 1 gaps." in result.stdout
    assert out_files(tmp_path) == TOKEN_FILES
    since = counts_since(site, before)
    assert since["/e1.html"] == 1 and since["/e2.html"] == 1
    assert since["/e3.html"] == 2  # position 3 has no file link, so it is a gap: full load
    assert since["/e4.html"] == 2 and since["/e6.html"] == 2
    assert json.loads((tmp_path / "state.json").read_text())["complete"] is True


def test_fill_gaps_downloads_only_gaps_and_report(tmp_path, site):
    site.broken.add(4)
    config = token_config(tmp_path, site, max_consecutive_failures=0)
    assert run(config).returncode == 0
    assert out_files(tmp_path) == ["1.jpg", "2.jpg", "6.jpg", "7.jpg"]
    report = run(config, "--gaps")
    assert report.returncode == 0
    assert "Furthest position: 6. Series complete: yes." in report.stdout
    assert "position 3: MAIN nofile" in report.stdout and "position 4: MAIN failed" in report.stdout

    site.broken.clear()
    site.token = "b2"
    before = dict(site.counts)
    result = run(config, "--fill-gaps", "--start-url", demo_site.entry_url(site))
    assert result.returncode == 0, result.stdout
    assert "[Page 4] Last gap processed. Stopping." in result.stdout
    assert out_files(tmp_path) == TOKEN_FILES
    since = counts_since(site, before)
    assert "/e5.html" not in since and "/e6.html" not in since
    assert [path for path in since if path.startswith("/files/")] == ["/files/n4"]
    assert json.loads((tmp_path / "state.json").read_text())["complete"] is True

    report = run(config, "--gaps")
    assert "position 4" not in report.stdout and "position 3: MAIN nofile" in report.stdout
    assert "MAIN numbers skipped by the site: 5" in report.stdout
    assert "MAIN numbers missing across other positions: 3" in report.stdout


def test_fill_gaps_without_gaps_does_nothing(tmp_path, site):
    config = token_config(tmp_path, site)
    assert run(config, "--max-pages", "2").returncode == 1
    before = dict(site.counts)
    result = run(config, "--fill-gaps")
    assert result.returncode == 0
    assert "No gaps up to position 2. Nothing to do." in result.stdout
    assert counts_since(site, before) == {}


def test_fill_gaps_halts_when_series_changed(tmp_path, site):
    site.broken.add(4)
    config = token_config(tmp_path, site, max_consecutive_failures=0)
    assert run(config).returncode == 0
    site.broken.clear()
    site.renumber[4] = 9
    site.token = "b2"
    result = run(config, "--fill-gaps", "--start-url", demo_site.entry_url(site))
    assert result.returncode == 1, result.stdout
    assert (
        "The series appears changed: MAIN file number 9 at position 4 is not between 2 and 6" in result.stdout
    )


# ==============================================================================
# TWO FILE STREAMS: MAIN AND SIDE
# ==============================================================================
SIDE_FILES = ["side1.pdf", "side4.pdf"]


def test_main_and_side_streams_into_separate_folders(tmp_path, site):
    config = token_config(tmp_path, site, side_file_link_pattern=demo_site.SIDE_PATTERN)
    result = run(config)
    assert result.returncode == 0, result.stdout
    assert out_files(tmp_path, "MAIN") == TOKEN_FILES
    assert out_files(tmp_path, "SIDE") == SIDE_FILES
    assert (tmp_path / "out" / "SIDE" / "side4.pdf").read_bytes() == b"SIDEDATAside4.pdf"
    rows = manifest_rows(tmp_path)
    side_saved = [r["saved_name"] for r in rows if r["stream"] == "SIDE" and r["status"] == "SAVE"]
    assert side_saved == ["SIDE/side1.pdf", "SIDE/side4.pdf"]
    # A page without a SIDE link is no gap.
    assert "Gaps: 1" in run(config, "--gaps").stdout  # only MAIN nofile at position 3

    # Rerun with new links: both streams are skipped by position, no file request.
    site.token = "b2"
    before = dict(site.counts)
    result = run(config, "--start-url", demo_site.entry_url(site))
    assert result.returncode == 0, result.stdout
    assert out_files(tmp_path, "SIDE") == SIDE_FILES
    assert not any(path.startswith("/files/") for path in counts_since(site, before))


def test_side_stream_off_without_pattern(tmp_path, site):
    config = token_config(tmp_path, site)
    assert run(config).returncode == 0
    assert out_files(tmp_path, "SIDE") == []  # the folder exists, but no SIDE pattern is set
    assert not any(path.startswith("/files/s") for path in site.counts)


def test_side_failure_is_filled_without_main_requests(tmp_path, site):
    site.side_broken.add(4)
    config = token_config(
        tmp_path, site, side_file_link_pattern=demo_site.SIDE_PATTERN, max_consecutive_failures=0
    )
    assert run(config).returncode == 0
    assert out_files(tmp_path, "SIDE") == ["side1.pdf"]
    assert "position 4: SIDE failed" in run(config, "--gaps").stdout

    site.side_broken.clear()
    site.token = "b2"
    before = dict(site.counts)
    result = run(config, "--fill-gaps", "--start-url", demo_site.entry_url(site))
    assert result.returncode == 0, result.stdout
    assert out_files(tmp_path, "SIDE") == SIDE_FILES
    assert out_files(tmp_path, "MAIN") == TOKEN_FILES
    since = counts_since(site, before)
    assert [path for path in since if path.startswith("/files/")] == ["/files/s4"]
