import os, sys, time, socket, threading, http.server, logging, re
from unittest.mock import patch
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class _NeverClosingMJPEGHandler(http.server.BaseHTTPRequestHandler):
    """Sends 200 + 1 KB for /video, then blocks forever (simulates an endless MJPEG stream)."""
    def do_GET(self):
        if self.path == "/video":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            self.wfile.write(b"--frame\r\n" + b"X" * 1024)
            self.wfile.flush()
            try:
                while True:
                    time.sleep(1)
            except Exception:
                pass
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok": true}')

    def log_message(self, *_):
        pass


def _start_mock_server(handler_cls=_NeverClosingMJPEGHandler, port=0):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
    actual_port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, actual_port, t


from wireless.laptop_server import LaptopMLServer


# ---- Test 1 ----------------------------------------------------------------
def test_startup_check_finishes_under_5s_with_hanging_video():
    """Startup check must finish in < 5 s even when /video never closes."""
    srv, port, _ = _start_mock_server(_NeverClosingMJPEGHandler)
    try:
        server = LaptopMLServer.__new__(LaptopMLServer)
        base = "http://127.0.0.1:%d" % port
        with patch("config.PI_HOST", "127.0.0.1"), \
             patch("config.PI_STREAM_URL", base + "/video"), \
             patch("config.SENSORS_URL",   base + "/sensors"), \
             patch("config.MOTION_URL",    base + "/motion"):
            t0 = time.monotonic()
            server._run_pi_startup_check()
            elapsed = time.monotonic() - t0
        assert elapsed < 5.0, "Blocked for %.1f s" % elapsed
    finally:
        srv.shutdown()


# ---- Test 2 ----------------------------------------------------------------
def test_startup_check_unreachable_host_does_not_block():
    """Connection-refused on a dead port must not block more than 5 s."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    dead_port = s.getsockname()[1]
    s.close()
    server = LaptopMLServer.__new__(LaptopMLServer)
    base = "http://127.0.0.1:%d" % dead_port
    with patch("config.PI_HOST", "127.0.0.1"), \
         patch("config.PI_STREAM_URL", base + "/video"), \
         patch("config.SENSORS_URL",   base + "/sensors"), \
         patch("config.MOTION_URL",    base + "/motion"):
        t0 = time.monotonic()
        server._run_pi_startup_check()
        elapsed = time.monotonic() - t0
    assert elapsed < 5.0, "Blocked for %.1f s" % elapsed


# ---- Test 3 ----------------------------------------------------------------
def test_startup_check_unresolvable_name_does_not_block():
    """An unresolvable hostname must not block the startup check."""
    server = LaptopMLServer.__new__(LaptopMLServer)
    fake = "this-host-does-not-exist-visionmate.local"
    base = "http://%s:5000" % fake
    with patch("config.PI_HOST", fake), \
         patch("config.PI_STREAM_URL", base + "/video"), \
         patch("config.SENSORS_URL",   base + "/sensors"), \
         patch("config.MOTION_URL",    base + "/motion"):
        t0 = time.monotonic()
        server._run_pi_startup_check()
        elapsed = time.monotonic() - t0
    assert elapsed < 5.0, "Blocked for %.1f s" % elapsed


# ---- Test 4 ----------------------------------------------------------------
def test_probe_url_closes_video_stream_after_1kb():
    """_probe_url(is_stream=True) must return True without reading the whole body."""
    srv, port, _ = _start_mock_server(_NeverClosingMJPEGHandler)
    try:
        url = "http://127.0.0.1:%d/video" % port
        t0 = time.monotonic()
        result = LaptopMLServer._probe_url(url, is_stream=True, total_timeout=3.0)
        elapsed = time.monotonic() - t0
        assert result is True, "Probe should return True for a 200 stream"
        assert elapsed < 3.5, "Stream not closed in time; took %.1f s" % elapsed
    finally:
        srv.shutdown()


# ---- Test 5 ----------------------------------------------------------------
def test_startup_check_logs_correct_summary(caplog):
    """
    _run_pi_startup_check must emit:
      Startup check (background): Pi hub OK|FAIL /video OK|FAIL /sensors OK|FAIL /motion OK|FAIL
    """
    server = LaptopMLServer.__new__(LaptopMLServer)
    fake = "this-host-does-not-exist-visionmate.local"
    with caplog.at_level(logging.INFO), \
         patch("config.PI_HOST", fake), \
         patch("config.PI_STREAM_URL", "http://%s:5000/video" % fake), \
         patch("config.SENSORS_URL",   "http://%s:5000/sensors" % fake), \
         patch("config.MOTION_URL",    "http://%s:5000/motion" % fake):
        server._run_pi_startup_check()
    matching = [r for r in caplog.records if "Startup check (background):" in r.getMessage()]
    assert matching, "No matching log line found"
    msg = matching[0].getMessage()
    for kw in ("Pi hub", "/video", "/sensors", "/motion"):
        assert kw in msg, "%r not in log: %r" % (kw, msg)


# ---- Test 6a ---------------------------------------------------------------
def test_no_hardcoded_ips_in_config():
    """config.py must not contain raw IPv4 literals."""
    ip_re = re.compile(r'\b(?:10|172|192)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
    with open(os.path.join(REPO_ROOT, "config.py")) as f:
        src = f.read()
    hits = ip_re.findall(src)
    assert not hits, "Hard-coded IPs in config.py: %s" % hits


# ---- Test 6b ---------------------------------------------------------------
def test_no_hardcoded_ips_in_startup_code():
    """wireless/laptop_server.py must not contain hard-coded Pi IPs."""
    ip_re = re.compile(r'\b(?:10|172|192)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
    with open(os.path.join(REPO_ROOT, "wireless", "laptop_server.py")) as f:
        src = f.read()
    hits = ip_re.findall(src)
    assert not hits, "Hard-coded IPs in laptop_server.py: %s" % hits


# ---- Test 7 ----------------------------------------------------------------
def test_probe_url_hard_timeout_via_future():
    """
    The hard total_timeout is enforced by Future.result(), not only by
    requests' socket timeout.  We simulate a server that stalls AFTER
    sending the status line.
    """
    class _SlowHeaderServer(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.wfile.write(b"HTTP/1.1 200 OK\r\n")
            self.wfile.flush()
            try:
                time.sleep(10)
            except Exception:
                pass

        def log_message(self, *_):
            pass

    srv, port, _ = _start_mock_server(_SlowHeaderServer)
    try:
        url = "http://127.0.0.1:%d/sensors" % port
        t0 = time.monotonic()
        result = LaptopMLServer._probe_url(url, is_stream=False, total_timeout=2.0)
        elapsed = time.monotonic() - t0
        assert elapsed < 4.0, "Hard timeout not enforced: %.1f s" % elapsed
        assert result is False, "Probe should FAIL when server stalls"
    finally:
        srv.shutdown()
