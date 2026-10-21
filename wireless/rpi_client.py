"""
Raspberry Pi Client for VisionMate Smart Glasses
Runs on: Raspberry Pi 4B (on the glasses)
Connects to: Laptop ML server via WiFi

Functions:
- Captures camera frames (30fps)
- Reads ESP32 sensor data (USB serial)
- Streams frames to laptop via WebSocket
- Receives ML results from laptop
- Plays audio instructions via Bluetooth
"""

import asyncio
import websockets
import cv2
import json
import base64
import time
import queue
import serial
import pyttsx3
from threading import Thread
import logging
import numpy as np
import os
import sys
from typing import Optional, Any

# Add parent directory to path to import VisionMate config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import config
except ImportError:
    config = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RaspberryPiClient:
    def __init__(
        self,
        server_ip="127.0.0.1",
        server_port=8081,
        client_speaks: Optional[bool] = None,
        camera_source: Optional[Any] = None,
    ):
        """
        Initialize Raspberry Pi client
        
        Args:
            server_ip: IP address of laptop running ML server (default: 127.0.0.1)
            server_port: WebSocket port (default 8081)
            client_speaks: Whether client plays speech (default: None, falls back to config.CLIENT_SPEAKS)
            camera_source: Optional camera index or stream URL
        """
        self.server_ip = server_ip
        self.server_port = server_port
        self.server_url = f"ws://{server_ip}:{server_port}"
        self.client_speaks = client_speaks
        self.camera_source = camera_source
        self.camera = None
        self.picamera2 = None
        self.use_picamera2 = False
        self.use_test_frames = False
        self.esp32_serial = None
        self.esp32_last_time = 0.0
        self.audio_engine = None
        self.audio_queue = queue.Queue()
        self.audio_worker_thread = None
        self.last_nav_announcement = ""
        self.last_nav_announcement_time = 0
        self.last_nav_log = ""
        self.last_nav_log_time = 0.0
        self.last_recognized_face_name = None
        self.last_recognized_face_log_time = 0.0
        self.running = False
        
        # Sensor data (None when ESP32 is not connected)
        self.ultrasonic_distance = None
        self.button_pressed = False
        self.distance_client = None
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()

    def _should_client_speak(self) -> bool:
        """Check whether client-side audio output is enabled"""
        if self.client_speaks is not None:
            return self.client_speaks
        if config is not None:
            return getattr(config, 'CLIENT_SPEAKS', False)
        return False
        
    def initialize(self, dry_run: bool = False):
        """Initialize all hardware components"""
        logger.info("Initializing Raspberry Pi Client...")
        
        if dry_run:
            self.use_test_frames = True
            # Verify test frame generation and JPEG encoding
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(test_frame, "DRY RUN", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            _, buffer = cv2.imencode('.jpg', test_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if buffer is None or len(buffer) == 0:
                raise RuntimeError("Failed to encode test frame during dry-run")
            logger.info("[OK] Camera initialized (dry run: test frame checks passed)")

            # Verify PIDistanceClient object instantiation without opening network connection
            from pi_distance_client import PIDistanceClient
            pi_host = getattr(config, 'PI_HOST', '10.117.98.217') if config else '10.117.98.217'
            self.distance_client = PIDistanceClient(host=pi_host)
            logger.info("[OK] ESP32 / Distance Client initialized (dry run: PIDistanceClient object created, network skipped)")

            # Audio engine check
            logger.info("[OK] Audio queue initialized (dry run: audio worker skipped)")
            logger.info("[OK] Raspberry Pi Client ready (dry run)!")
            return True
        
        # Initialize camera - try multiple methods
        camera_initialized = False
        
        # Method 1: Try picamera2 (for new Raspberry Pi OS with libcamera)
        try:
            from picamera2 import Picamera2
            logger.info("Trying picamera2...")
            self.picamera2 = Picamera2()
            cam_config = self.picamera2.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            self.picamera2.configure(cam_config)
            self.picamera2.start()
            time.sleep(2)  # Let camera warm up (2s for reliability)
            
            # Test capture
            test_frame = self.picamera2.capture_array()
            if test_frame is not None and test_frame.shape[0] > 0:
                self.use_picamera2 = True
                camera_initialized = True
                logger.info(f"[OK] Camera initialized with picamera2 (frame: {test_frame.shape})")
            else:
                logger.warning("picamera2 started but capture returned empty frame, closing...")
                try:
                    self.picamera2.stop()
                    self.picamera2.close()
                except Exception:
                    pass
                self.picamera2 = None
        except Exception as e:
            logger.warning(f"picamera2 failed: {e}")
            # Clean up any partially initialized picamera2 to release camera resource
            if self.picamera2 is not None:
                try:
                    self.picamera2.stop()
                except Exception:
                    pass
                try:
                    self.picamera2.close()
                except Exception:
                    pass
                self.picamera2 = None
        
        # Method 2: Use open_camera helper (network stream or local camera)
        if not camera_initialized:
            try:
                from camera_source import open_camera
                cam_source = self.camera_source
                if cam_source is None and config is not None:
                    cam_source = getattr(config, 'CAMERA_SOURCE', 0)
                logger.info(f"Opening camera source: {cam_source}...")
                self.camera = open_camera(cam_source, width=640, height=480, fps=30)
                ret, test_frame = self.camera.read()
                if ret and test_frame is not None and test_frame.size > 0:
                    camera_initialized = True
                    logger.info(f"[OK] Camera initialized with open_camera (frame: {test_frame.shape})")
                else:
                    logger.warning("open_camera opened but could not read frame")
                    self.camera.release()
                    self.camera = None
            except Exception as e:
                logger.warning(f"open_camera failed: {e}")
                if self.camera is not None:
                    try:
                        self.camera.release()
                    except Exception:
                        pass
                    self.camera = None

        # Method 4: Use test frames for demo (if no camera available)
        if not camera_initialized:
            logger.warning("No camera available, using TEST FRAMES for demo")
            self.use_test_frames = True
            camera_initialized = True
        
        if not camera_initialized:
            logger.error("Camera initialization failed!")
            return False
        
        # Initialize ESP32 serial connection
        try:
            # Try common USB serial ports
            for port in ['/dev/ttyUSB0', '/dev/ttyACM0', '/dev/ttyUSB1']:
                try:
                    self.esp32_serial = serial.Serial(port, 115200, timeout=0.1)
                    logger.info(f"[OK] ESP32 connected on {port}")
                    break
                except:
                    continue
            
            if not self.esp32_serial:
                logger.info("Local ESP32 serial not found; starting PIDistanceClient to poll Pi distance endpoint...")
                try:
                    from pi_distance_client import PIDistanceClient
                    pi_host = getattr(config, 'PI_HOST', '10.117.98.217') if config else '10.117.98.217'
                    self.distance_client = PIDistanceClient(host=pi_host).start()
                except Exception as e:
                    logger.warning(f"Could not start PIDistanceClient: {e}")
        except Exception as e:
            logger.warning(f"ESP32 connection failed: {e}")
            try:
                from pi_distance_client import PIDistanceClient
                pi_host = getattr(config, 'PI_HOST', '10.117.98.217') if config else '10.117.98.217'
                self.distance_client = PIDistanceClient(host=pi_host).start()
            except Exception:
                pass
        
        # Start persistent audio worker thread
        self.audio_worker_thread = Thread(target=self._audio_worker, daemon=True)
        self.audio_worker_thread.start()
        logger.info("[OK] Audio engine initialized (Bluetooth output)")
        
        logger.info("[OK] Raspberry Pi Client ready!")
        return True
    
    def _audio_worker(self):
        """Dedicated worker thread for processing audio playback queue"""
        import sys
        if sys.platform == 'win32':
            import pythoncom
            pythoncom.CoInitialize()
        
        try:
            try:
                self.audio_engine = pyttsx3.init()
                self.audio_engine.setProperty('rate', 180)  # Speed
                self.audio_engine.setProperty('volume', 1.0)  # Volume
            except Exception as e:
                logger.warning(f"Audio engine init failed on worker thread: {e}")
                self.audio_engine = None

            while True:
                text = self.audio_queue.get()
                if self.audio_engine:
                    try:
                        self.audio_engine.say(text)
                        self.audio_engine.runAndWait()
                    except Exception as e:
                        logger.error(f"Audio playback error: {e}")
                        if getattr(self.audio_engine, '_inLoop', False):
                            try:
                                self.audio_engine.endLoop()
                            except Exception:
                                self.audio_engine._inLoop = False
        finally:
            if sys.platform == 'win32':
                try:
                    import pythoncom
                    pythoncom.CoUninitialize()
                except Exception:
                    pass
    
    def read_esp32_sensors(self):
        """Read sensor data from ESP32 via PIDistanceClient or local serial"""
        if getattr(self, 'distance_client', None) is not None:
            val = self.distance_client.get_distance()
            self.ultrasonic_distance = val
            if val is not None:
                self.esp32_last_time = time.time()
            return

        if not self.esp32_serial:
            # No mock distances: send None when ESP32 is not found
            self.ultrasonic_distance = None
            return
        
        try:
            if self.esp32_serial.in_waiting > 0:
                line = self.esp32_serial.readline().decode('utf-8', errors='ignore').strip()
                
                # Parse: "DIST:123.45,BTN:0"
                if line.startswith("DIST:"):
                    parts = line.split(',')
                    
                    # Parse distance
                    dist_str = parts[0].split(':')[1]
                    val = float(dist_str)
                    if 2.0 <= val <= 400.0:
                        self.ultrasonic_distance = val
                        self.esp32_last_time = time.time()
                    
                    # Parse button
                    if len(parts) > 1:
                        btn_str = parts[1].split(':')[1]
                        self.button_pressed = (btn_str == '1')
        except Exception as e:
            logger.debug(f"ESP32 read error: {e}")
    
    def speak(self, text):
        """Queue audio for playback (handled by the dedicated audio worker thread)"""
        if not self._should_client_speak():
            return
        if text:
            self.audio_queue.put(text)
    
    async def capture_and_stream(self, websocket):
        """Main loop: capture frames and stream to laptop"""
        logger.info("Starting video streaming...")
        recv_task = asyncio.create_task(self._recv_loop(websocket))
        
        while self.running:
            try:
                # Capture frame based on available method
                frame = None
                
                if self.use_picamera2:
                    # Method 1: picamera2
                    frame = self.picamera2.capture_array()
                    # Convert RGB to BGR for OpenCV compatibility
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                elif self.camera is not None:
                    # Method 2: OpenCV
                    ret, frame = self.camera.read()
                    if not ret:
                        logger.error("Failed to capture frame")
                        await asyncio.sleep(0.1)
                        continue
                        
                elif self.use_test_frames:
                    # Method 3: Generate test frame
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    frame[:] = (50, 100, 150)  # BGR color
                    # Add text showing it's a test
                    cv2.putText(frame, "TEST MODE - Camera Not Available", (50, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    import random
                    # Simulate some objects
                    cv2.rectangle(frame, (100, 100), (200, 200), (0, 255, 0), 2)
                    cv2.putText(frame, f"Object {random.randint(1,5)}", (105, 95),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                if frame is None:
                    await asyncio.sleep(0.1)
                    continue
                
                # Read ESP32 sensors
                self.read_esp32_sensors()
                
                # Encode frame as JPEG (85% quality for speed)
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Check staleness: None when unavailable or older than 500 ms
                now = time.time()
                dist_val = self.ultrasonic_distance
                if dist_val is not None:
                    if (now - getattr(self, 'esp32_last_time', 0.0)) > 0.5:
                        dist_val = None

                # Prepare data packet
                data = {
                    'frame': frame_base64,
                    'timestamp': now,
                    'distance': dist_val,
                    'ultrasonic': dist_val,
                    'button': self.button_pressed,
                    'frame_shape': frame.shape[:2]  # (height, width)
                }
                
                # Send to laptop
                await websocket.send(json.dumps(data))
                
                # Track FPS
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed
                    dist_str = f"{self.ultrasonic_distance:.1f}cm" if self.ultrasonic_distance is not None else "None"
                    logger.info(f"Streaming @ {fps:.1f} fps, Distance: {dist_str}")
                
                # Control frame rate (target 30fps)
                await asyncio.sleep(0.001)
                
            except websockets.exceptions.ConnectionClosed:
                logger.error("Connection to server lost")
                break
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                await asyncio.sleep(0.1)
        recv_task.cancel()
        try:
            await recv_task
        except (asyncio.CancelledError, Exception):
            pass

    async def _recv_loop(self, websocket):
        """Asynchronously receive and process ML results from laptop without blocking frame transmission."""
        try:
            async for response in websocket:
                if not self.running:
                    break
                try:
                    result = json.loads(response)
                    self.process_results(result)
                except Exception as e:
                    logger.debug(f"Error processing response: {e}")
        except (websockets.exceptions.ConnectionClosedOK,
                websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosed):
            pass
        except Exception as e:
            logger.debug(f"Receiver loop exception: {e}")

    
    def process_results(self, result):
        """Process ML results from laptop"""
        try:
            # Get navigation instruction
            if 'navigation' in result and result['navigation']:
                nav_text = result['navigation']
                now = time.time()
                
                # Only log "Navigation: <msg>" when the message changes or 10 seconds have passed
                if nav_text != self.last_nav_log or (now - self.last_nav_log_time) >= 10.0:
                    logger.info(f"Navigation: {nav_text}")
                    self.last_nav_log = nav_text
                    self.last_nav_log_time = now
                
                # Only speak server guidance/messages through client audio engine when CLIENT_SPEAKS is True
                if self._should_client_speak():
                    if nav_text != self.last_nav_announcement or (now - self.last_nav_announcement_time) > 2.0:
                        self.speak(nav_text)
                        self.last_nav_announcement = nav_text
                        self.last_nav_announcement_time = now
            
            # Log detections
            if 'objects' in result:
                obj_count = len(result['objects'])
                if obj_count > 0:
                    logger.debug(f"Detected {obj_count} objects")
            
            if 'faces' in result:
                face_count = len(result['faces'])
                if face_count > 0:
                    names = [f['name'] for f in result['faces']]
                    names_str = ', '.join(names)
                    now_face = time.time()
                    if names_str != self.last_recognized_face_name or (now_face - self.last_recognized_face_log_time) >= 10.0:
                        logger.info(f"Recognized: {names_str}")
                        self.last_recognized_face_name = names_str
                        self.last_recognized_face_log_time = now_face
                else:
                    self.last_recognized_face_name = None
            
            # Calculate latency
            if 'timestamp' in result:
                latency = (time.time() - result['timestamp']) * 1000
                logger.debug(f"Round-trip latency: {latency:.1f}ms")
                
        except Exception as e:
            logger.error(f"Result processing error: {e}")
    
    async def connect_and_run(self):
        """Connect to laptop server and start streaming"""
        self.running = True
        
        print(f"Connecting to laptop at: {self.server_ip} ({self.server_url})")
        logger.info(f"Connecting to laptop server: {self.server_url}")
        
        try:
            async with websockets.connect(self.server_url, ping_interval=None, open_timeout=20, close_timeout=10) as websocket:
                logger.info("[OK] Connected to laptop ML server!")
                
                # Start streaming
                await self.capture_and_stream(websocket)
                
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            logger.error("Make sure laptop server is running and IP address is correct!")
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        
        if self.picamera2:
            try:
                self.picamera2.stop()
                logger.info("picamera2 stopped")
            except:
                pass
        
        if self.camera:
            self.camera.release()
            logger.info("Camera released")
        
        if getattr(self, 'distance_client', None) is not None:
            try:
                self.distance_client.stop()
                logger.info("PIDistanceClient stopped")
            except Exception:
                pass

        if self.esp32_serial:
            self.esp32_serial.close()
            logger.info("ESP32 disconnected")
        
        logger.info("Raspberry Pi Client stopped")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="VisionMate Raspberry Pi Client")
    parser.add_argument("laptop_ip", nargs="?", default="127.0.0.1", help="Laptop IP address")
    parser.add_argument("--source", default=None, help="Camera source (plain number for webcam index, or stream URL)")
    parser.add_argument("--dry-run", action="store_true", help="Run startup checks, print 'DRY RUN OK' and exit without opening camera, mic, or network")
    args = parser.parse_args()

    laptop_ip = args.laptop_ip
    cam_source = args.source
    if cam_source is None:
        cam_source = os.environ.get("VISIONMATE_CAMERA")

    if cam_source is not None and cam_source.isdigit():
        cam_source = int(cam_source)

    if args.dry_run:
        client = RaspberryPiClient(server_ip=laptop_ip, server_port=8081, camera_source=cam_source)
        ok = client.initialize(dry_run=True)
        client.cleanup()
        if ok:
            print("DRY RUN OK")
            sys.exit(0)
        else:
            logger.error("Dry run initialization failed!")
            sys.exit(1)

    print(f"Connecting to laptop at: {laptop_ip}")
    if cam_source is not None:
        print(f"Using camera source: {cam_source}")
    logger.info(f"Connecting to laptop at: {laptop_ip}")

    # Create and initialize client
    client = RaspberryPiClient(server_ip=laptop_ip, server_port=8081, camera_source=cam_source)

    if not client.initialize():
        logger.error("Initialization failed!")
        return

    # Run client
    try:
        asyncio.run(client.connect_and_run())
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    finally:
        client.cleanup()


if __name__ == "__main__":
    main()
