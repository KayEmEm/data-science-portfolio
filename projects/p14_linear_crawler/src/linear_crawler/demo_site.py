"""
Synthetic page series on 127.0.0.1 for the tests and the smoke run.

Every page and file is generated here; no real site is involved. The page
layout matches the example patterns in config.yaml: a file link with the text
"Download file" and a next page link with rel="next".
"""

import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

FILE_PATTERN = r'href="([^"]*)"(?=[^>]*>Download file<)'
NEXT_PATTERN = r'href="([^"]*)"[^>]*rel="next"'
COOKIE_NAME = "session"
COOKIE_VALUE = "demo"


def nav(next_url):
    return f'<img src="/logo.png"> <a href="/home">Home</a> <a href="{next_url}" rel="next">Next</a>'


def file_link(url):
    return f'<a class="file" href="{url}">Download file</a>'


PAGES = {
    # Main series: 4 pages, page 2 has no file, page 4 links to itself.
    "/p1.html": nav("p2.html") + "\n" + file_link("files/one.jpg?x=1&amp;y=2"),
    "/p2.html": nav("/p3.html"),
    "/p3.html": file_link("files/cd") + "\n" + nav("p4.html#top"),
    "/p4.html": file_link("files/one.jpg?y=2") + "\n" + nav("p4.html"),
    # Special cases.
    "/empty.html": "<html><body>nothing</body></html>",
    "/h1.html": file_link("files/html") + "\n" + nav("h2.html"),
    "/h2.html": file_link("files/html") + "\n" + nav("h2.html"),
    "/r1.html": file_link("files/limited") + "\n" + nav("r1.html"),
    "/login.html": "<html><body>Please sign in</body></html>",
    "/s1.html": file_link("files/login") + "\n" + nav("s1.html"),
    "/c1.html": file_link("files/short") + "\n" + nav("c1.html"),
    "/j1.html": file_link("files/one.jpg?y=2")
    + "\n"
    + nav("j1.html")
    + '\n<script>location.replace("/elsewhere.html")</script>',
    "/m1.html": '<meta http-equiv="refresh" content="0;url=/elsewhere.html">\n'
    + file_link("files/one.jpg?y=2")
    + "\n"
    + nav("m1.html"),
    "/t1.html": file_link("files/one.jpg?y=2")
    + "\n"
    + nav("t1.html")
    + '\n<script>setTimeout(() => { location.href = "/elsewhere.html"; }, 0)</script>',
    "/elsewhere.html": "<html><body>other page</body></html>",
}

# Series with expiring links: pages /e1.html to /e6.html, valid only with the
# current token (?t=...). File links carry the same token. Files are named by
# Content-Disposition with an unpadded number. Position 3 has no file, and the
# site skips number 5. server.token changes the token, which expires every
# page and file link. server.broken lists file numbers that answer HTTP 500;
# server.expire_once lists file numbers whose next request answers with an
# "Invalid link" page; server.renumber maps a file number to another number.
TOKEN_NUMBERS = {1: 1, 2: 2, 3: None, 4: 4, 5: 6, 6: 7}
INVALID_LINK = b"<html><body>Invalid link</body></html>"


def token_page(position, token):
    last = len(TOKEN_NUMBERS)
    following = position if position == last else position + 1
    body = nav(f"e{following}.html?t={token}")
    number = TOKEN_NUMBERS[position]
    if number is not None:
        body += "\n" + file_link(f"files/n{number}?t={token}")
    return body.encode()


def entry_url(server):
    """Entry URL of the token series with the current token."""
    return f"{server.base}/e1.html?t={server.token}"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        server = self.server
        url = urlsplit(self.path)
        cookie = self.headers.get("Cookie", "")
        with server.lock:
            server.requests.append(url.path)
            server.user_agents.append(self.headers.get("User-Agent", ""))
            server.counts[url.path] = server.counts.get(url.path, 0) + 1
            count = server.counts[url.path]

        if url.path == "/old.html":
            return self.reply(302, b"", "text/plain", {"Location": "/login.html"})
        if url.path == "/p4.html" and url.query == "go=1":
            return self.reply(302, b"", "text/plain", {"Location": "/p4.html"})
        if url.path in PAGES:
            return self.reply(200, PAGES[url.path].encode(), "text/html")
        token = parse_qs(url.query).get("t", [""])[0]
        match = re.fullmatch(r"/e(\d+)\.html", url.path)
        if match and int(match.group(1)) in TOKEN_NUMBERS:
            if token != server.token:
                return self.reply(200, INVALID_LINK, "text/html")
            return self.reply(200, token_page(int(match.group(1)), token), "text/html")
        match = re.fullmatch(r"/files/n(\d+)", url.path)
        if match:
            number = int(match.group(1))
            with server.lock:
                expire_now = number in server.expire_once
                server.expire_once.discard(number)
            if token != server.token or expire_now:
                return self.reply(200, INVALID_LINK, "text/html")
            if number in server.broken:
                return self.reply(500, b"", "text/plain")
            name = f"{server.renumber.get(number, number)}.jpg"
            disposition = {"Content-Disposition": f'attachment; filename="{name}"'}
            return self.reply(200, b"FILEDATA" + name.encode(), "image/jpeg", disposition)
        if not url.path.startswith("/files/"):
            return self.reply(404, b"", "text/plain")

        # Files need the HttpOnly session cookie from the cookie file.
        if f"{COOKIE_NAME}={COOKIE_VALUE}" not in cookie:
            return self.reply(403, b"", "text/plain")
        if url.path == "/files/one.jpg" and parse_qs(url.query).get("y") != ["2"]:
            return self.reply(400, b"", "text/plain")
        if url.path == "/files/html":
            return self.reply(200, b"<html>Error page</html>", "text/html")
        if url.path == "/files/login":
            return self.reply(200, b"<html>Please sign in</html>", "text/html")
        if url.path == "/files/limited" and count == 1:
            return self.reply(429, b"", "text/plain", {"Retry-After": "1"})
        if url.path == "/files/short":
            return self.reply(200, b"abc", "image/jpeg", {"Content-Length": "10"}, set_length=False)
        headers = {}
        if url.path == "/files/cd":
            headers["Content-Disposition"] = "attachment; filename*=UTF-8''r%C3%A9sum%C3%A9.jpg"
        return self.reply(200, b"FILEDATA" + url.path.encode(), "image/jpeg", headers)

    def reply(self, status, body, ctype, headers=None, set_length=True):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        if set_length:
            self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def start():
    """Starts the site on a free port in a background thread. Returns the server."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.lock = threading.Lock()
    server.requests = []
    server.user_agents = []
    server.counts = {}
    server.token = "a1"
    server.broken = set()
    server.expire_once = set()
    server.renumber = {}
    server.base = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def stop(server):
    server.shutdown()
    server.server_close()


def write_cookie_file(path):
    """Netscape cookie file with one session cookie and one HttpOnly cookie."""
    path.write_text(
        "# Netscape HTTP Cookie File\n"
        "127.0.0.1\tFALSE\t/\tFALSE\t0\tsess\tabc\n"
        f"#HttpOnly_127.0.0.1\tFALSE\t/\tFALSE\t4000000000\t{COOKIE_NAME}\t{COOKIE_VALUE}\n",
        encoding="utf-8",
    )
    return path
