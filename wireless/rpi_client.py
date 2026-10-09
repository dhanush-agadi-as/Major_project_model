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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RaspberryPiClient:
    def __init__(self, server_ip="192.168.1.100", server_port=8081):
        """
        Initialize Raspberry Pi client
        
        Args:
            server_ip: IP address of laptop running ML server
            server_port: WebSocket port (default 8081)
        """
        self.server_url = f"ws://{server_ip}:{server_port}"
        self.camera = None
        self.picamera2 = None
        self.use_picamera2 = False
        self.use_test_frames = False
        self.esp32_serial = None
        self.audio_engine = None
        self.audio_queue = queue.Queue()
        self.audio_worker_thread = None
        self.last_nav_announcement = ""
        self.last_nav_announcement_time = 0
        self.running = False
        
        # Sensor data
        self.ultrasonic_distance = 999.0
        self.button_pressed = False
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
    def initialize(self):
        """Initialize all hardware components"""
        logger.info("Initializing Raspberry Pi Client...")
        
        # Initialize camera - try multiple methods
        camera_initialized = False
        
        # Method 1: Try picamera2 (for new Raspberry Pi OS with libcamera)
        try:
            from picamera2 import Picamera2
            logger.info("Trying picamera2...")
            self.picamera2 = Picamera2()
            config = self.picamera2.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            self.picamera2.configure(config)
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
        
        # Method 2: Try OpenCV with V4L2 (for older Raspberry Pi OS)
        if not camera_initialized:
            try:
                logger.info("Trying OpenCV V4L2...")
                self.camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera.set(cv2.CAP_PROP_FPS, 30)
                
                ret, test_frame = self.camera.read()
                if ret and test_frame is not None:
                    camera_initialized = True
                    logger.info(f"[OK] Camera initialized with OpenCV (frame: {test_frame.shape})")
                else:
                    logger.warning("OpenCV V4L2 opened but could not read frame")
                    self.camera.release()
                    self.camera = None
            except Exception as e:
                logger.warning(f"OpenCV camera failed: {e}")
                if self.camera is not None:
                    try:
                        self.camera.release()
                    except Exception:
                        pass
                    self.camera = None

        # Method 3: OpenCV with DSHOW (Windows laptop testing fallback).
        # CAP_V4L2 above is Linux-only (Raspberry Pi); on Windows it "opens"
        # without erroring but can never read a real frame. DSHOW is the
        # correct backend for a Windows laptop webcam used during local dev.
        if not camera_initialized:
            try:
                logger.info("Trying OpenCV DSHOW (Windows)...")
                self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera.set(cv2.CAP_PROP_FPS, 30)

                ret, test_frame = self.camera.read()
                if ret and test_frame is not None:
                    camera_initialized = True
                    logger.info(f"[OK] Camera initialized with OpenCV DSHOW (frame: {test_frame.shape})")
                else:
                    logger.warning("OpenCV DSHOW opened but could not read frame")
                    self.camera.release()
                    self.camera = None
            except Exception as e:
                logger.warning(f"OpenCV DSHOW camera failed: {e}")
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
                logger.warning("ESP32 not found, using mock sensor data")
        except Exception as e:
            logger.warning(f"ESP32 connection failed: {e}, using mock data")
        
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
        """Read sensor data from ESP32 via serial"""
        if not self.esp32_serial:
            # Mock data for testing
            import random
            self.ultrasonic_distance = random.uniform(30, 200)
            return
        
        try:
            if self.esp32_serial.in_waiting > 0:
                line = self.esp32_serial.readline().decode('utf-8', errors='ignore').strip()
                
                # Parse: "DIST:123.45,BTN:0"
                if line.startswith("DIST:"):
                    parts = line.split(',')
                    
                    # Parse distance
                    dist_str = parts[0].split(':')[1]
                    self.ultrasonic_distance = float(dist_str)
                    
                    # Parse button
                    if len(parts) > 1:
                        btn_str = parts[1].split(':')[1]
                        self.button_pressed = (btn_str == '1')
        except Exception as e:
            logger.debug(f"ESP32 read error: {e}")
    
    def speak(self, text):
        """Queue audio for playback (handled by the dedicated audio worker thread)"""
        if text:
            self.audio_queue.put(text)
    
    async def capture_and_stream(self, websocket):
        """Main loop: capture frames and stream to laptop"""
        logger.info("Starting video streaming...")
        
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
                
                # Prepare data packet
                data = {
                    'frame': frame_base64,
                    'timestamp': time.time(),
                    'ultrasonic': self.ultrasonic_distance,
                    'button': self.button_pressed,
                    'frame_shape': frame.shape[:2]  # (height, width)
                }
                
                # Send to laptop
                await websocket.send(json.dumps(data))
                
                # Wait for response from laptop
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                    result = json.loads(response)
                    
                    # Process results
                    self.process_results(result)
                    
                except asyncio.TimeoutError:
                    logger.debug("No response from server (timeout)")
                
                # Track FPS
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed
                    logger.info(f"Streaming @ {fps:.1f} fps, Distance: {self.ultrasonic_distance:.1f}cm")
                
                # Control frame rate (target 30fps)
                await asyncio.sleep(0.001)
                
            except websockets.exceptions.ConnectionClosed:
                logger.error("Connection to server lost")
                break
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                await asyncio.sleep(0.1)
    
    def process_results(self, result):
        """Process ML results from laptop"""
        try:
            # Get navigation instruction
            if 'navigation' in result and result['navigation']:
                nav_text = result['navigation']
                logger.info(f"Navigation: {nav_text}")
                
                # Speak instruction with cooldown to avoid queuing the same message every frame
                now = time.time()
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
                    logger.info(f"Recognized: {', '.join(names)}")
            
            # Calculate latency
            if 'timestamp' in result:
                latency = (time.time() - result['timestamp']) * 1000
                logger.debug(f"Round-trip latency: {latency:.1f}ms")
                
        except Exception as e:
            logger.error(f"Result processing error: {e}")
    
    async def connect_and_run(self):
        """Connect to laptop server and start streaming"""
        self.running = True
        
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
        
        if self.esp32_serial:
            self.esp32_serial.close()
            logger.info("ESP32 disconnected")
        
        logger.info("Raspberry Pi Client stopped")


def main():
    """Main entry point"""
    import sys
    
    # Get laptop IP from command line or use default
    if len(sys.argv) > 1:
        laptop_ip = sys.argv[1]
    else:
        laptop_ip = "192.168.1.100"  # CHANGE THIS TO YOUR LAPTOP IP
        print(f"\nUsage: python3 rpi_client.py <LAPTOP_IP>")
        print(f"Using default IP: {laptop_ip}")
        print(f"To find laptop IP: On laptop run 'ipconfig' (Windows) or 'ifconfig' (Linux/Mac)\n")
    
    # Create and initialize client
    client = RaspberryPiClient(server_ip=laptop_ip, server_port=8081)
    
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
