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
        logger.info(
            "IPGeolocationTracker initialized "
            f"(cache={cache_seconds}s, endpoint={IP_API_URL})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_position(self) -> Optional[Tuple[float, float]]:
        """
        Return (latitude, longitude) from ip-api.com.

        Returns a cached value if still within cache_seconds; otherwise
        re-fetches with retry + exponential backoff.  After MAX_CONSECUTIVE_FAILURES
        consecutive network errors the tracker enters a cooldown period
        (FAILURE_COOLDOWN_SECONDS) before trying again, preventing request storms
        on a bad network connection.

        Returns None on network error or API failure.
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
        """Try _fetch_position up to _MAX_RETRIES+1 times with exponential backoff."""
        for attempt in range(self._MAX_RETRIES + 1):
            result = self._fetch_position()
            if result is not None:
                return result
            if attempt < self._MAX_RETRIES:
                delay = self._RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    f"IPGeolocationTracker: attempt {attempt + 1} failed, "
                    f"retrying in {delay:.0f}s ..."
                )
                time.sleep(delay)
        # All attempts exhausted
        self._consecutive_failures += 1
        if self._consecutive_failures == self._MAX_CONSECUTIVE_FAILURES:
            self._failure_start_time = time.time()
            logger.error(
                f"IPGeolocationTracker: {self._consecutive_failures} consecutive "
                f"failures -- entering {self._FAILURE_COOLDOWN_SECONDS}s cooldown"
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

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_HEADERS = {"User-Agent": "VisionMate-HAL102/1.0"}


def reverse_geocode(lat: float, lon: float, cache_seconds: float = 120.0) -> "Optional[str]":
    """
    Convert GPS coordinates to a human-readable place name using
    OpenStreetMap Nominatim (free, no API key required).

    Args:
        lat: Latitude
        lon: Longitude
        cache_seconds: How long (seconds) to reuse a cached result for the
                       same rounded coordinate (default 120 s = 2 min).

    Returns:
        Short place description string (e.g. "MG Road, Bangalore") or None
        on any network / parse failure. Callers should treat None as
        "(unknown location)" -- no exception is ever raised.

    IMPORTANT -- Nominatim usage policy:
        Max 1 request/second. This function caches aggressively (keyed by
        coordinates rounded to 4 decimal places, ~11 m precision) and is
        only called when a new sighting is recorded (which already has its own
        cooldown via OBJECT_MATCH_COOLDOWN), so real-world call rate is far
        below the 1 req/s limit.
    """
    if lat is None or lon is None:
        return None

    # Round to ~4 decimal places (~11 m precision) for cache key
    rounded_lat = round(lat, 4)
    rounded_lon = round(lon, 4)
    cache_key = (rounded_lat, rounded_lon)

    now = time.time()
    if cache_key in _reverse_geocode_cache:
        cached_time, cached_name = _reverse_geocode_cache[cache_key]
        if (now - cached_time) < cache_seconds:
            return cached_name

    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"format": "json", "lat": lat, "lon": lon, "zoom": 18},
            headers=NOMINATIM_HEADERS,
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()

        address = data.get("address", {})
        parts = []
        for field in ("road", "suburb", "city_district", "city", "town", "village"):
            val = address.get(field)
            if val:
                parts.append(val)
                if len(parts) >= 2:
                    break

        if parts:
            place = ", ".join(parts)
        else:
            place = data.get("display_name", "").split(",")[0].strip() or None

        _reverse_geocode_cache[cache_key] = (now, place)
        logger.debug(f"Reverse geocode ({lat:.4f}, {lon:.4f}) -> {place!r}")
        return place

    except Exception as exc:
        logger.debug(f"reverse_geocode failed for ({lat}, {lon}): {exc}")
        return None


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
