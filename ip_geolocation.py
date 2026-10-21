"""
IP-Based Geolocation Fallback
Uses ip-api.com -- 100% free, no sign-up, no API key required.

ACCURACY CAVEAT:
    This is city-level accuracy only (~1-10 km typical error).
    It is NOT a substitute for real GPS hardware.
    Accuracy degrades significantly on VPNs, mobile data, or shared NAT.
    Use only for development/testing when no GPS hardware is available.

RATE LIMIT (free tier, non-commercial):
    45 requests per minute per IP, HTTP only (not HTTPS on free tier).
    This class caches results for `cache_seconds` (default 5 min) to stay
    well within that limit during normal VisionMate sessions.

Legal: http://ip-api.com/docs/legal -- non-commercial use only on free tier.
"""

import logging
import time
import threading
import requests
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

IP_API_URL = "http://ip-api.com/json/"  # HTTP only on free tier -- do not change to HTTPS


class IPGeolocationTracker:
    """
    Provides approximate (lat, lon) by querying the public ip-api.com endpoint.

    Drop-in compatible with GPSTracker.get_position() -- both return
    Optional[Tuple[float, float]].

    Args:
        cache_seconds: How long (in seconds) to reuse a cached lookup before
                       re-querying. Default 300 s (5 min) keeps well within
                       ip-api.com's 45 req/min rate limit.
    """

    def __init__(self, cache_seconds: float = 300.0):
        self._cache_seconds = cache_seconds
        self._cached_position: Optional[Tuple[float, float]] = None
        self._cached_city: Optional[str] = None
        self._last_fetch_time: float = 0.0
        self._consecutive_failures: int = 0
        self._failure_start_time: float = 0.0
        self.is_running: bool = False
        self.geolocation_thread: Optional[threading.Thread] = None
        logger.info(
            "IPGeolocationTracker initialized "
            f"(cache={cache_seconds}s, endpoint={IP_API_URL})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """
        Start background geolocation updates in a daemon thread.
        Does not start at all when GPS_MOCK_MODE or INDOOR_MODE is True.
        """
        try:
            import config
            mock_mode = getattr(config, 'GPS_MOCK_MODE', False)
            indoor_mode = getattr(config, 'INDOOR_MODE', False)
        except Exception:
            mock_mode = False
            indoor_mode = False

        if mock_mode or indoor_mode:
            logger.info("IPGeolocationTracker: not started because GPS_MOCK_MODE or INDOOR_MODE is True")
            return

        if self.is_running:
            return

        self.is_running = True
        self.geolocation_thread = threading.Thread(
            target=self._run_loop,
            name="IPGeolocationThread",
            daemon=True
        )
        self.geolocation_thread.start()
        logger.info("IPGeolocationTracker background thread started (daemon=True)")

    def stop(self):
        """Stop background geolocation updates."""
        self.is_running = False
        self.geolocation_thread = None

    def _run_loop(self):
        """Background update loop running in daemon thread."""
        while self.is_running:
            try:
                self.get_position()
            except Exception as e:
                logger.error(f"Error in IPGeolocationThread: {e}")
            sleep_chunks = max(1, int(self._cache_seconds))
            for _ in range(sleep_chunks):
                if not self.is_running:
                    break
                time.sleep(1.0)

    def get_position(self) -> Optional[Tuple[float, float]]:
        """
        Return (latitude, longitude) from ip-api.com.

        Returns a cached value if still within cache_seconds; otherwise
        fetches position. If the fetch fails (e.g. no internet), enters the
        600s cooldown immediately without a retry loop.
        """
        if self._is_cache_fresh():
            return self._cached_position

        # Honour give-up cooldown after repeated failures
        if self._in_failure_cooldown():
            logger.debug(
                f"IPGeolocationTracker: in failure cooldown "
                f"({self._consecutive_failures} failures), skipping fetch"
            )
            return self._cached_position  # may still be a stale-but-valid position

        return self._fetch_with_retry()

    # constants for retry/backoff policy
    _MAX_RETRIES = 2                  # extra attempts beyond the first (3 total)
    _RETRY_BASE_DELAY = 2.0           # seconds before first retry
    _MAX_CONSECUTIVE_FAILURES = 5     # give up after this many consecutive errors
    _FAILURE_COOLDOWN_SECONDS = 600   # 10 min cooldown after max failures reached

    def get_city(self) -> Optional[str]:
        """
        Return the last successfully resolved city as "City, Region", or None.
        Does NOT trigger a new network request -- uses whatever is cached.
        Call get_position() first to ensure the cache is populated.
        """
        return self._cached_city

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_cache_fresh(self) -> bool:
        """True if the cached result is still within cache_seconds."""
        if self._cached_position is None:
            return False
        age = time.time() - self._last_fetch_time
        return age < self._cache_seconds

    def _in_failure_cooldown(self) -> bool:
        """True if we are currently in the give-up cooldown period."""
        if self._consecutive_failures < self._MAX_CONSECUTIVE_FAILURES:
            return False
        elapsed = time.time() - self._failure_start_time
        if elapsed >= self._FAILURE_COOLDOWN_SECONDS:
            # Cooldown expired -- reset and allow a fresh attempt
            logger.info("IPGeolocationTracker: failure cooldown expired, will retry")
            self._consecutive_failures = 0
            self._failure_start_time = 0.0
            return False
        return True

    def _fetch_with_retry(self) -> Optional[Tuple[float, float]]:
        """
        Query _fetch_position once.
        When the first attempt fails because of no internet / error,
        enter the 600 s cooldown immediately without a retry loop.
        """
        result = self._fetch_position()
        if result is not None:
            return result

        # First attempt failed -- enter 600s cooldown immediately (no retry loop)
        self._consecutive_failures = self._MAX_CONSECUTIVE_FAILURES
        self._failure_start_time = time.time()
        logger.error(
            f"IPGeolocationTracker: Fetch failed (no internet) -- "
            f"entering {self._FAILURE_COOLDOWN_SECONDS}s cooldown immediately (no retry loop)"
        )
        return None

    def _fetch_position(self) -> Optional[Tuple[float, float]]:
        """
        Single attempt to query ip-api.com/json/ and cache the result.

        Returns (lat, lon) on success, None on any failure.
        Errors are logged clearly but never raised -- callers should treat
        None as 'no position available right now, retry later'.
        """
        try:
            logger.info(f"Fetching IP geolocation from {IP_API_URL} ...")
            response = requests.get(IP_API_URL, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "success":
                reason = data.get("message", "unknown")
                logger.warning(f"ip-api.com returned non-success status: {reason}")
                return None

            lat = float(data["lat"])
            lon = float(data["lon"])
            city = data.get("city", "Unknown")
            region = data.get("regionName", "Unknown")
            city_str = f"{city}, {region}"

            self._cached_position = (lat, lon)
            self._cached_city = city_str
            self._last_fetch_time = time.time()
            self._consecutive_failures = 0  # reset on success

            logger.info(
                f"IP geolocation resolved: {lat:.4f}, {lon:.4f} -- {city_str} "
                f"(city-level accuracy only, may be off by several km)"
            )
            return self._cached_position

        except requests.exceptions.ConnectionError:
            logger.error(
                "IPGeolocationTracker: No internet connection -- "
                "cannot reach ip-api.com. Returning None."
            )
            return None
        except requests.exceptions.Timeout:
            logger.error(
                "IPGeolocationTracker: Request to ip-api.com timed out (10 s). "
                "Returning None."
            )
            return None
        except Exception as e:
            logger.error(f"IPGeolocationTracker: Unexpected error during fetch: {e}")
            return None

# ---------------------------------------------------------------------------
# Reverse Geocoding  (lat/lon -> human-readable place name)
# ---------------------------------------------------------------------------

# Module-level cache: maps (rounded_lat, rounded_lon) -> (timestamp, place_name)
_reverse_geocode_cache: dict = {}
_reverse_geocode_lock = threading.Lock()
_last_reverse_geocode_time: float = 0.0

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_HEADERS = {"User-Agent": "VisionMate/1.0"}


def clear_reverse_geocode_cache() -> None:
    """Clear the reverse geocode cache (useful for testing)."""
    global _last_reverse_geocode_time
    with _reverse_geocode_lock:
        _reverse_geocode_cache.clear()
        _last_reverse_geocode_time = 0.0


def reverse_geocode(lat: float, lon: float, cache_seconds: float = 300.0) -> "Optional[str]":
    """
    Convert GPS coordinates to a human-readable place name using
    OpenStreetMap Nominatim (free, no API key required).

    Args:
        lat: Latitude
        lon: Longitude
        cache_seconds: Cache duration in seconds (default 300 s = 5 minutes).

    Returns:
        Short place description: road + suburb/neighbourhood + city (e.g. "Brigade Road, Ashok Nagar, Bengaluru")
        or None on failure. Callers can fall back to IP city or 'your current location'.
    """
    global _last_reverse_geocode_time
    if lat is None or lon is None:
        return None

    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (ValueError, TypeError):
        return None

    # Cache by lat/lon rounded to 4 decimals (~11m precision) for 5 minutes
    rounded_lat = round(lat_f, 4)
    rounded_lon = round(lon_f, 4)
    cache_key = (rounded_lat, rounded_lon)

    now = time.time()
    with _reverse_geocode_lock:
        if cache_key in _reverse_geocode_cache:
            cached_time, cached_name = _reverse_geocode_cache[cache_key]
            if (now - cached_time) < cache_seconds:
                return cached_name

        # Enforce max 1 request per second
        time_since_last = now - _last_reverse_geocode_time
        if time_since_last < 1.0:
            time.sleep(1.0 - time_since_last)
        _last_reverse_geocode_time = time.time()

    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"format": "jsonv2", "lat": lat_f, "lon": lon_f, "zoom": 18},
            headers=NOMINATIM_HEADERS,
            timeout=3,
        )
        resp.raise_for_status()
        data = resp.json()

        address = data.get("address", {})
        parts = []

        # 1. Road
        road = (
            address.get("road")
            or address.get("pedestrian")
            or address.get("footway")
            or address.get("path")
            or address.get("street")
        )
        if road:
            parts.append(road.strip())

        # 2. Suburb or neighbourhood
        suburb = (
            address.get("suburb")
            or address.get("neighbourhood")
            or address.get("residential")
            or address.get("subdistrict")
            or address.get("quarter")
        )
        if suburb and suburb.strip() not in parts:
            parts.append(suburb.strip())

        # 3. City
        city = (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("municipality")
            or address.get("city_district")
        )
        if city and city.strip() not in parts:
            parts.append(city.strip())

        if parts:
            place = ", ".join(parts)
        else:
            display_name = data.get("display_name", "")
            place = display_name.split(",")[0].strip() if display_name else None

        with _reverse_geocode_lock:
            _reverse_geocode_cache[cache_key] = (time.time(), place)
        logger.debug(f"Reverse geocode ({lat_f:.4f}, {lon_f:.4f}) -> {place!r}")
        return place

    except Exception as exc:
        logger.debug(f"reverse_geocode failed for ({lat}, {lon}): {exc}")
        return None


def get_place_name(lat: Optional[float], lon: Optional[float], city_fallback: Optional[str] = None) -> str:
    """
    Get human-readable place name.
    Falls back to IP-geolocation city if reverse_geocode fails, then 'your current location'.
    Never returns raw numbers or coordinates.
    """
    if lat is not None and lon is not None:
        try:
            name = reverse_geocode(lat, lon)
            if name:
                return name
        except Exception:
            pass

    if city_fallback and str(city_fallback).strip():
        return str(city_fallback).strip()

    return "your current location"


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("\n" + "=" * 60)
    print("  IPGeolocationTracker -- Standalone Test")
    print("=" * 60)
    print("  Service  : ip-api.com (free, no API key)")
    print("  Accuracy : CITY-LEVEL only (~1-10 km typical error)")
    print("             Worse on VPN / mobile data / shared NAT.")
    print("             This is a testing fallback, NOT real GPS.")
    print("=" * 60)

    tracker = IPGeolocationTracker(cache_seconds=300.0)

    print("\nFetching position (first call -- hits the network)...")
    pos = tracker.get_position()

    if pos:
        lat, lon = pos
        city = tracker.get_city()
        print(f"\n  Latitude  : {lat:.6f}")
        print(f"  Longitude : {lon:.6f}")
        print(f"  City      : {city}")
        print(f"\n  ACCURACY NOTE: The coordinates above represent the")
        print(f"  approximate centre of your ISP's city/region, NOT")
        print(f"  your actual physical location. Expect errors of")
        print(f"  1-10 km or more in typical conditions.")

        print("\nFetching again (should use cache -- no network request)...")
        pos2 = tracker.get_position()
        assert pos2 == pos, "Cache mismatch!"
        print("  Cache hit confirmed -- ip-api.com was NOT contacted again.")
    else:
        print("\n  Could not fetch position.")
        print("  Check your internet connection and try again.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  Test complete!")
    print("=" * 60 + "\n")
