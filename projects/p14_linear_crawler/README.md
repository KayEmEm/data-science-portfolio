# Project 14: Linear crawler

Follow a series of web pages in which each page links to one file and to the next page, save every file, and stop at the last page, which links to itself. The crawler drives a real Chromium browser with [Playwright](https://playwright.dev/python/), reuses the cookies of a logged in browser session, and is built to run unattended through long series.

## Headline result

**On a synthetic series, the crawler saves every file, recovers from the failures that stop simple download scripts, and halts with a clear reason when it cannot continue. 48 tests cover these cases; `make smoke` runs the full crawl in about 2 seconds.** The tests run the crawler as a separate process against [`demo_site.py`](src/linear_crawler/demo_site.py), a local site on 127.0.0.1 that produces each situation on purpose.

| Situation | Behavior | Test |
| --- | --- | --- |
| Page without a file link | logged, page recorded as `MISS`, crawl continues | `test_full_series_then_rerun_skips` |
| Neither link on a page | halt, page source saved for debugging | `test_both_links_absent_halts` |
| Next link points to the current page | end of series, exit code 0 | `test_full_series_then_rerun_skips` |
| Next link points to an earlier page | cycle detected, halt | no dedicated test |
| HTTP 429 with `Retry-After` | waits the requested time, then retries | `test_retry_after_is_used` |
| File response is an HTML page | not saved, counted as a failure; halt after N in a row | `test_html_file_response_and_consecutive_failure_halt` |
| Login text or redirect to another path | session treated as expired, exit code 4 | `test_session_expired_pattern`, `test_redirect_to_other_path_halts` |
| Truncated transfer | retried, then recorded as `FAIL` | `test_content_length_mismatch_is_retried_then_fails` |
| Page navigates away by itself (script redirect, meta refresh) | source read with a direct request; no hang | `test_page_that_navigates_by_itself` |
| Browser closed or crashed | browser restarted, page retried, up to a limit | `test_browser_lost_restart` |
| Rerun over saved files | files skipped by the manifest | `test_full_series_then_rerun_skips` |
| Interruption | state saved; `--resume` continues at that page | `test_max_pages_then_resume` |
| Escape key, Ctrl+C, SIGTERM | stop with summary; terminal settings restored | `test_escape_key_stops_and_arrow_key_does_not`, `test_sigterm_writes_summary` |
| Second run on the same folder | refused by a lock, exit code 3 | `test_lock_blocks_second_run` |

## Data and license

This project has no dataset. It collects files from a site that the person running it can open in their own browser. Whoever runs it is responsible for the site's terms of use and for a request rate the site accepts; the default delays (10 to 25 seconds between pages) are set for that purpose. The repository contains no real site, URL, cookie or link pattern: the tests and the smoke run use only the synthetic site, and the patterns in `config.yaml` are examples that match it.

Code: MIT, as the rest of the repository.

## How it works

1. **Page visit.** Chromium opens the page, so that the page scripts run and the cookies stay current. Stylesheets, images, fonts and media can be blocked (`block_subresources`), because only the HTML is used.
2. **Page source.** The raw HTML comes from a second, direct request with the browser cookies. The source of the browser response is not used: Chromium discards it when a page navigates away by itself right after loading, and reading it can then fail or hang.
3. **Links.** Two regular expressions from `config.yaml` find the file link and the next page link in the raw HTML, line by line (`re.MULTILINE`), with the URL in capture group 1. HTML entities are decoded and relative URLs resolved. The first match is used, and more than one match is logged.
4. **File.** The file is fetched with the browser cookies, by Playwright's HTTP client or, with `stream_downloads`, streamed to disk by the standard library. The file name comes from `Content-Disposition` or the URL; an existing name gets a suffix (`one_1.jpg`). Each file is written to a temporary name first, then renamed, and recorded in `manifest.csv` with its size and SHA256 hash.
5. **Next page.** The crawl stops when the next link points to the current page (end of series), to an earlier page (cycle), or is missing, or when a limit is reached (`max_pages`, consecutive failures, browser restarts).
6. **Failures.** Each page load and download has up to `max_download_retries` retries. The wait doubles from `min_firewall_sleep` up to `max_firewall_sleep`, with random variation, unless the server sends `Retry-After`. HTTP 403, 429 and 503 are logged as possible blocks.

The browser runs in one of three display modes (`display_mode`): `visible` (a normal window), `headless` (no window; Chromium's newer headless mode) or `virtual` (a normal window on a private Xvfb display, not visible on screen). `virtual` behaves like `visible` and needs the `xvfb` package.

## How to reproduce

```
make install        # this package and Chromium for Playwright
make smoke          # full crawl of the synthetic site, outputs under artifacts/smoke
make test           # 48 tests
```

To crawl a real series:

1. Export the cookies of the logged in browser in Netscape format to `data/raw/cookies.txt` (the `data/` folder is never committed).
2. In `config.yaml`, set `start_url`, the two link patterns, and `user_agent`: the "User agent" line from `chrome://version` in the Chrome browser that exported the cookies. The crawler refuses to start without it.
3. `make dry-run ARGS="--max-pages 3"` shows the links the patterns find; `make crawl` starts the crawl and `make resume` continues an interrupted one.

Several series can run at the same time, each with its own config file, if each uses its own `download_dir`, `state_file`, `manifest_file`, `crawl_log` and `debug_dir`.

## Settings

All settings are in [`config.yaml`](config.yaml).

| Group | Keys |
| --- | --- |
| Series and patterns | `start_url`, `file_link_pattern`, `next_page_pattern`, `max_pages` |
| Session | `cookie_file`, `user_agent`, `session_expired_pattern`, `halt_on_redirect`, `cookie_expiry_warn_hours` |
| Browser | `display_mode`, `block_subresources`, `context_restart_pages`, `max_browser_restarts`, `timeout_seconds` |
| Pace and retries | `min_page_delay`, `max_page_delay`, `max_download_retries`, `min_firewall_sleep`, `max_firewall_sleep`, `max_retry_after_seconds`, `max_consecutive_failures` |
| Files | `download_dir`, `stream_downloads`, `reject_html_files`, `skip_saved` |
| Records | `manifest_file`, `state_file`, `debug_dir`, `crawl_log`, `console_level`, `log_max_bytes`, `log_backups` |
| Control | `escape_key_stop` |

Command line: `python -m linear_crawler <config> [--start-url URL | --resume] [--max-pages N] [--dry-run]`.

| Exit code | Meaning |
| ---: | --- |
| 0 | series complete |
| 1 | halted before the end (reason in the log) |
| 2 | usage or configuration error |
| 3 | another run holds the lock on the download folder |
| 4 | session expired |
| 130 | Ctrl+C or Escape key |
| 143 | SIGTERM |

## Departures from the shared conventions

* **No dataset, model or metric.** The Makefile has `crawl`, `resume`, `dry-run` and `smoke` targets instead of `data`, `train` and `evaluate`, and there is no MLflow tracking or baseline. `sample` runs the smoke crawl, so that the Colab profiles can run this project like the others; `SAMPLE` has no effect.
* **Own settings loader.** `config.yaml` is read by the crawler itself, not by `portfolio_common.config`: `--print-config` and `PORTFOLIO_CONFIG` are not supported, and relative paths are relative to the folder the crawler starts from. The Makefile targets run from this folder.
* **Linux.** The crawler imports `fcntl` and `termios` for the folder lock and the Escape key, so it needs Linux or macOS, and the `virtual` display mode needs Xvfb. It is tested on Linux only.

## Limits

* Links are found in the raw HTML. Links that page scripts add later are not seen.
* Each page is requested twice, once by the browser and once for its source.
* A site can tell a headless browser from a visible one; `virtual` avoids that difference.
* Playwright's Chromium reports its own version in the `Sec-CH-UA` headers, which can differ from the version in `user_agent`.
