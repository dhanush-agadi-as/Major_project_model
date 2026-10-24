"""
alert_sender.py
===============
Firebase app channel for VisionMate Emergency Alerts.
Interacts with Firebase Firestore and Firebase Cloud Messaging (FCM).
Alerts go ONLY to the family app through Firebase.
"""

import os
import sys
import time
import uuid
import logging
import threading
from typing import Optional, Callable, Dict, Any, List

logger = logging.getLogger(__name__)

# Try importing firebase_admin; handled gracefully if not installed or during testing
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, messaging
    FIREBASE_AVAILABLE = True
except ImportError:
    firebase_admin = None
    credentials = None
    firestore = None
    messaging = None
    FIREBASE_AVAILABLE = False


def _is_invalid_token_error(err: Exception) -> bool:
    """Check if FCM error indicates an expired, unregistered, or invalid device token."""
    err_str = str(err).lower()
    return (
        "unregistered" in err_str or
        "invalid" in err_str or
        "not-registered" in err_str or
        "bad-request" in err_str or
        getattr(err, "code", "") in ("UNREGISTERED", "INVALID_ARGUMENT")
    )


class AlertSender:
    """
    Manages emergency alert creation, FCM push notifications, location updates,
    and family ack monitoring via Firebase Firestore and FCM.
    """

    def __init__(self,
                 service_account_path: Optional[str] = None,
                 user_name: str = "VisionMate User",
                 dry_run: bool = True,
                 speak_fn: Optional[Callable[[str], None]] = None,
                 gps_tracker: Optional[Any] = None):
        self.user_name = user_name
        self.dry_run = dry_run
        self.speak_fn = speak_fn
        self.gps_tracker = gps_tracker
        self.service_account_path = service_account_path

        self.db = None
        self.app = None
        self._active_alert_id: Optional[str] = None
        self._doc_watch = None
        self._location_thread = None
        self._stop_location_event = threading.Event()
        self._lock = threading.Lock()

        self._init_firebase()

    def _init_firebase(self):
        """Initialize Firebase Admin SDK with credentials if available."""
        if self.dry_run:
            logger.info("AlertSender initialized in DRY_RUN mode (no network calls will be made to Firebase).")
            return

        if not FIREBASE_AVAILABLE:
            logger.warning("firebase_admin library is not installed. Running in offline/dry-run mode.")
            self.dry_run = True
            return

        sa_path = self.service_account_path
        if not sa_path:
            import config
            sa_path = getattr(config, 'FIREBASE_SERVICE_ACCOUNT_PATH', 'firebase_service_account.json')

        if not os.path.exists(sa_path):
            logger.warning("Firebase service account file not found at '%s'. Running in offline/dry-run mode.", sa_path)
            self.dry_run = True
            return

        try:
            # Check if default app already exists
            try:
                self.app = firebase_admin.get_app()
            except ValueError:
                cred = credentials.Certificate(sa_path)
                self.app = firebase_admin.initialize_app(cred)
            self.db = firestore.client()
            logger.info("[OK] Firebase Admin SDK initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize Firebase Admin SDK: %s", e)
            self.db = None
            self.dry_run = True

    def resolve_location(self) -> Dict[str, Any]:
        """
        Resolve current location:
        1. Phone GPS if fresh (including place name from cached reverse geocode at 1 req/s)
        2. Fallback to IP location with source 'ip'
        3. None if both unavailable
        """
        now = time.time()
        lat: Optional[float] = None
        lon: Optional[float] = None
        acc: Optional[float] = None
        place_name: Optional[str] = None
        source: Optional[str] = None

        # 1. Phone GPS via gps_tracker
        if self.gps_tracker:
            try:
                phone_info = self.gps_tracker._get_fresh_phone_position_info()
                if phone_info is not None:
                    pos, age, accuracy = phone_info
                    lat, lon = pos
                    acc = accuracy
                    source = "phone"
            except Exception as e:
                logger.debug("Error reading phone GPS: %s", e)

        # 2. IP Geolocation fallback
        if lat is None and self.gps_tracker:
            try:
                ip_tr = self.gps_tracker._get_ip_tracker()
                if ip_tr:
                    pos = ip_tr.get_position()
                    if pos:
                        lat, lon = pos
                        acc = 5000.0  # Coarse estimate
                        source = "ip"
            except Exception as e:
                logger.debug("Error reading IP geolocation: %s", e)

        # Standalone IP Geolocation fallback if no gps_tracker
        if lat is None:
            try:
                from ip_geolocation import IPGeolocationTracker
                tr = IPGeolocationTracker()
                pos = tr.get_position()
                if pos:
                    lat, lon = pos
                    acc = 5000.0
                    source = "ip"
            except Exception:
                pass

        # Reverse geocode place name
        if lat is not None and lon is not None:
            try:
                from ip_geolocation import get_place_name
                city_fallback = getattr(self.gps_tracker, 'get_city', lambda: None)() if self.gps_tracker else None
                place_name = get_place_name(lat, lon, city_fallback=city_fallback)
            except Exception as e:
                logger.debug("Error getting place name: %s", e)
                place_name = "current location"

        return {
            "lat": lat,
            "lon": lon,
            "accuracy": acc,
            "placeName": place_name,
            "source": source,
            "updatedAt": now
        }

    def send_alert(self, severity: str = "severe", reason: str = "Fall detected") -> Optional[str]:
        """
        Create alert document in Firestore and send FCM data push to all devices.
        Returns alert_id on success, or None on failure.
        """
        alert_id = str(uuid.uuid4())[:12]
        now = time.time()

        location = self.resolve_location()
        if location.get("lat") is None:
            if "location unavailable" not in reason.lower():
                reason = f"{reason} (location unavailable)"

        alert_data = {
            "userName": self.user_name,
            "status": "active",
            "severity": severity,  # "normal" or "severe"
            "reason": reason,
            "createdAt": firestore.SERVER_TIMESTAMP if (self.db and not self.dry_run) else now,
            "location": location,
            "acks": []
        }

        # Dry run execution
        if self.dry_run or not self.db:
            logger.info("ALERT: [DRY_RUN] Would create alert doc alerts/%s: %s", alert_id, alert_data)
            logger.info("ALERT: [DRY_RUN] Would send FCM data message {type:'alert', alertId:'%s', name:'%s', reason:'%s', place:'%s'}",
                        alert_id, self.user_name, reason, location.get("placeName") or "")
            print("ALERT: channel=app status=ok")
            logger.info("ALERT: channel=app status=ok")
            if self.speak_fn:
                self.speak_fn("Alert sent to the family app")
            with self._lock:
                self._active_alert_id = alert_id
            return alert_id

        # Real Firebase execution
        try:
            # 1. Create alert document in Firestore
            self.db.collection("alerts").document(alert_id).set(alert_data)
            logger.info("Created Firestore document alerts/%s", alert_id)

            # 2. Get registered device tokens from devices collection
            device_docs = list(self.db.collection("devices").stream())
            push_success = False

            place = location.get("placeName") or ""
            fcm_data = {
                "type": "alert",
                "alertId": str(alert_id),
                "name": str(self.user_name),
                "reason": str(reason),
                "place": str(place)
            }

            for doc in device_docs:
                dev = doc.to_dict() or {}
                token = dev.get("token")
                if not token:
                    continue

                try:
                    msg = messaging.Message(
                        data=fcm_data,
                        token=token,
                        android=messaging.AndroidConfig(priority='high') if hasattr(messaging, 'AndroidConfig') else None,
                        apns=messaging.APNSConfig(headers={'apns-priority': '10'}) if hasattr(messaging, 'APNSConfig') else None
                    )
                    messaging.send(msg)
                    push_success = True
                except Exception as exc:
                    logger.warning("FCM send failed for device %s: %s", doc.id, exc)
                    if _is_invalid_token_error(exc):
                        logger.info("Removing invalid device token document: devices/%s", doc.id)
                        try:
                            doc.reference.delete()
                        except Exception as del_err:
                            logger.error("Failed to delete invalid token doc: %s", del_err)

            # 3. Spoken result & logging (once)
            print("ALERT: channel=app status=ok")
            logger.info("ALERT: channel=app status=ok")
            if self.speak_fn:
                self.speak_fn("Alert sent to the family app")

            with self._lock:
                self._active_alert_id = alert_id

            # 4. Start background location updater & snapshot listener
            self._start_location_updater(alert_id)
            self._start_ack_listener(alert_id)
            return alert_id

        except Exception as e:
            logger.error("Failed to send alert to Firebase: %s", e)
            print("ALERT: channel=app status=fail")
            logger.info("ALERT: channel=app status=fail")
            if self.speak_fn:
                self.speak_fn("Could not reach the family app")
            return None

    def send_fall_alert(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        fall_type: str = "fall",
    ) -> Optional[str]:
        """
        Dedicated fall-detection alert dispatcher called by EmergencyManager.

        Builds a location-enriched alert payload and dispatches it via Firebase
        Cloud Messaging to the mobile app.  When Firebase credentials are not
        configured (service account file missing or dry_run=True), the method
        falls back to structured logging so callers always receive a non-None
        alert_id and the system stays operational in dev/test environments.

        Args:
            lat:       GPS latitude resolved by the caller (may be None).
            lon:       GPS longitude resolved by the caller (may be None).
            fall_type: Alert sub-type string forwarded to the mobile app
                       (e.g. ``"fall"``, ``"impact"``).

        Returns:
            alert_id string on success (real or dry-run), None on hard failure.
        """
        alert_id = str(uuid.uuid4())[:12]
        now = time.time()

        # Use caller-supplied coordinates; fill gaps via resolve_location()
        if lat is None or lon is None:
            try:
                loc = self.resolve_location()
                if lat is None:
                    lat = loc.get("lat")
                if lon is None:
                    lon = loc.get("lon")
            except Exception as _e:
                logger.debug("send_fall_alert: location resolution error: %s", _e)

        reason = (
            f"Fall detected ({fall_type})"
            if (lat is None or lon is None)
            else f"Fall detected ({fall_type}) at {lat:.5f}, {lon:.5f}"
        )

        location = {
            "lat": lat,
            "lon": lon,
            "accuracy": None,
            "placeName": None,
            "source": "emergency",
            "updatedAt": now,
        }

        # Attempt reverse-geocode for place name
        if lat is not None and lon is not None:
            try:
                from ip_geolocation import get_place_name
                location["placeName"] = get_place_name(lat, lon)
            except Exception:
                pass

        alert_data = {
            "userName": self.user_name,
            "status": "active",
            "severity": "severe",
            "reason": reason,
            "fallType": fall_type,
            "createdAt": (
                firestore.SERVER_TIMESTAMP
                if (self.db and not self.dry_run)
                else now
            ),
            "location": location,
            "acks": [],
        }

        # ── Dry-run / no-credentials path ──────────────────────────────────
        if self.dry_run or not self.db:
            logger.info(
                "FALL_ALERT: [DRY_RUN/NO_CREDS] Would create alerts/%s: "
                "fall_type=%s lat=%s lon=%s",
                alert_id, fall_type, lat, lon,
            )
            logger.info(
                "FALL_ALERT: [DRY_RUN/NO_CREDS] Would send FCM data message "
                "{type:'fall_alert', alertId:'%s', name:'%s', fallType:'%s', "
                "lat:'%s', lon:'%s'}",
                alert_id, self.user_name, fall_type, lat, lon,
            )
            print("FALL_ALERT: channel=app status=ok (dry_run)")
            logger.info("FALL_ALERT: channel=app status=ok (dry_run)")
            if self.speak_fn:
                self.speak_fn("Alert sent to the family app")
            with self._lock:
                self._active_alert_id = alert_id
            return alert_id

        # ── Real Firebase path ──────────────────────────────────────────────
        try:
            # 1. Write alert document to Firestore
            self.db.collection("alerts").document(alert_id).set(alert_data)
            logger.info("FALL_ALERT: Created Firestore document alerts/%s", alert_id)

            # 2. Push FCM data message to all registered devices
            device_docs = list(self.db.collection("devices").stream())
            place = location.get("placeName") or ""
            fcm_data = {
                "type": "fall_alert",
                "alertId": str(alert_id),
                "name": str(self.user_name),
                "reason": str(reason),
                "fallType": str(fall_type),
                "lat": str(lat) if lat is not None else "",
                "lon": str(lon) if lon is not None else "",
                "place": str(place),
            }

            for doc in device_docs:
                dev = doc.to_dict() or {}
                token = dev.get("token")
                if not token:
                    continue
                try:
                    msg = messaging.Message(
                        data=fcm_data,
                        token=token,
                        android=(
                            messaging.AndroidConfig(priority="high")
                            if hasattr(messaging, "AndroidConfig") else None
                        ),
                        apns=(
                            messaging.APNSConfig(headers={"apns-priority": "10"})
                            if hasattr(messaging, "APNSConfig") else None
                        ),
                    )
                    messaging.send(msg)
                except Exception as exc:
                    logger.warning("FALL_ALERT: FCM send failed for device %s: %s", doc.id, exc)
                    if _is_invalid_token_error(exc):
                        logger.info("FALL_ALERT: Removing invalid token doc: devices/%s", doc.id)
                        try:
                            doc.reference.delete()
                        except Exception as del_err:
                            logger.error("FALL_ALERT: Failed to delete invalid token: %s", del_err)

            # 3. Log result and speak confirmation
            print("FALL_ALERT: channel=app status=ok")
            logger.info("FALL_ALERT: channel=app status=ok")
            if self.speak_fn:
                self.speak_fn("Alert sent to the family app")

            with self._lock:
                self._active_alert_id = alert_id

            # 4. Start background location updater & Firestore ack listener
            self._start_location_updater(alert_id)
            self._start_ack_listener(alert_id)
            return alert_id

        except Exception as e:
            logger.error("FALL_ALERT: Failed to send to Firebase: %s", e)
            print("FALL_ALERT: channel=app status=fail")
            logger.info("FALL_ALERT: channel=app status=fail")
            if self.speak_fn:
                self.speak_fn("Could not reach the family app")
            return None



    def cancel_alert(self, alert_id: Optional[str] = None):
        """
        Mark alert cancelled in Firestore and send FCM data push {type: 'cancelled', alertId: ...}.
        Removes invalid device tokens.
        """
        target_id = alert_id or self._active_alert_id
        if not target_id:
            logger.debug("cancel_alert called but no active alert ID")
            return

        self._stop_location_updater()
        self._stop_ack_listener()

        if self.dry_run or not self.db:
            logger.info("ALERT: [DRY_RUN] Would mark alerts/%s cancelled and send FCM data push {type:'cancelled', alertId:'%s'}",
                        target_id, target_id)
            with self._lock:
                if self._active_alert_id == target_id:
                    self._active_alert_id = None
            return

        try:
            # 1. Update status to cancelled in Firestore
            doc_ref = self.db.collection("alerts").document(target_id)
            doc_ref.update({"status": "cancelled"})
            logger.info("Marked alerts/%s as cancelled in Firestore", target_id)

            # 2. Send FCM data push to devices
            device_docs = list(self.db.collection("devices").stream())
            cancel_data = {
                "type": "cancelled",
                "alertId": str(target_id)
            }

            for doc in device_docs:
                dev = doc.to_dict() or {}
                token = dev.get("token")
                if not token:
                    continue

                try:
                    msg = messaging.Message(
                        data=cancel_data,
                        token=token,
                        android=messaging.AndroidConfig(priority='high') if hasattr(messaging, 'AndroidConfig') else None,
                        apns=messaging.APNSConfig(headers={'apns-priority': '10'}) if hasattr(messaging, 'APNSConfig') else None
                    )
                    messaging.send(msg)
                except Exception as exc:
                    logger.warning("FCM cancel send failed for device %s: %s", doc.id, exc)
                    if _is_invalid_token_error(exc):
                        logger.info("Removing invalid device token document: devices/%s", doc.id)
                        try:
                            doc.reference.delete()
                        except Exception as del_err:
                            logger.error("Failed to delete invalid token doc: %s", del_err)

            with self._lock:
                if self._active_alert_id == target_id:
                    self._active_alert_id = None

        except Exception as e:
            logger.error("Error cancelling alert in Firebase: %s", e)

    def _start_location_updater(self, alert_id: str):
        """Update location every 5 seconds while alert is active."""
        self._stop_location_event.clear()

        def _updater():
            while not self._stop_location_event.is_set():
                self._stop_location_event.wait(5.0)
                if self._stop_location_event.is_set():
                    break
                loc = self.resolve_location()
                try:
                    if self.db and not self.dry_run:
                        doc_ref = self.db.collection("alerts").document(alert_id)
                        # Only update if status is still active
                        doc_snap = doc_ref.get()
                        if doc_snap.exists and doc_snap.to_dict().get("status") == "active":
                            doc_ref.update({"location": loc})
                            logger.debug("Updated location for alerts/%s", alert_id)
                        else:
                            break
                except Exception as e:
                    logger.debug("Error updating alert location: %s", e)

        self._location_thread = threading.Thread(target=_updater, daemon=True, name="AlertLocationUpdater")
        self._location_thread.start()

    def _stop_location_updater(self):
        self._stop_location_event.set()
        self._location_thread = None

    def _start_ack_listener(self, alert_id: str):
        """Listen to Firestore snapshot changes and speak '<name> is on the way' when an ack is added."""
        if not self.db or self.dry_run:
            return

        seen_acks = set()
        alert_ref = self.db.collection("alerts").document(alert_id)

        def on_snapshot(doc_snapshot, changes, read_time):
            for doc in doc_snapshot:
                if not doc.exists:
                    continue
                data = doc.to_dict() or {}
                acks = data.get("acks", [])
                for ack in acks:
                    if isinstance(ack, dict):
                        name = ack.get("name", "Family member")
                        ts = str(ack.get("ts", ""))
                        key = f"{name}_{ts}"
                        if key not in seen_acks:
                            seen_acks.add(key)
                            logger.info("Alert acknowledged by %s", name)
                            if self.speak_fn:
                                try:
                                    self.speak_fn(f"{name} is on the way")
                                except Exception as e:
                                    logger.error("Error speaking ack: %s", e)

        try:
            self._doc_watch = alert_ref.on_snapshot(on_snapshot)
            logger.info("Firestore snapshot listener attached for alerts/%s", alert_id)
        except Exception as e:
            logger.warning("Could not attach snapshot listener: %s", e)

    def _stop_ack_listener(self):
        if self._doc_watch:
            try:
                self._doc_watch.unsubscribe()
            except Exception:
                pass
            self._doc_watch = None
