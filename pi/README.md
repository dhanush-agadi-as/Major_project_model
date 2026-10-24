# Raspberry Pi Hub (VisionMate)

The Raspberry Pi Hub runs `pi/pi_hub.py` to stream video, ultrasonic sensor data, and IMU motion telemetry for fall detection.

## MPU6050 I2C Hardware Setup & Wiring

The MPU6050 6-axis accelerometer & gyroscope connects to the Raspberry Pi GPIO header over I2C.

### Pin Wiring

| MPU6050 Pin | Raspberry Pi Pin | Description |
|-------------|------------------|-------------|
| **VCC**     | Pin 1 (3.3V)     | 3.3V Power  |
| **GND**     | Pin 6 (GND)      | Ground      |
| **SDA**     | Pin 3 (GPIO 2)   | I2C Data    |
| **SCL**     | Pin 5 (GPIO 3)   | I2C Clock   |

> **Note**: Connect VCC to 3.3V (do not connect to 5V unless using a 5V breakout board with an onboard level shifter).

---

## Software Configuration

### 1. Enable I2C Interface

Run raspi-config on the Raspberry Pi:
```bash
sudo raspi-config
```
Navigate to:
`Interface Options` -> `I2C` -> Select `Yes` to enable the ARM I2C interface.

Alternatively, enable it via command line:
```bash
sudo raspi-config nonint do_i2c 0
```

### 2. Install Required Packages

Install I2C diagnostics and python smbus package:
```bash
sudo apt update
sudo apt install -y i2c-tools python3-smbus python3-smbus2
```

### 3. Verify I2C Connection

Detect connected devices on I2C bus 1:
```bash
i2cdetect -y 1
```

The output grid should show `68` at row `60:`, column `8` (hex address `0x68`):
```text
     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
00:                         -- -- -- -- -- -- -- -- 
10: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
20: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
30: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
40: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
50: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
60: -- -- -- -- -- -- -- -- 68 -- -- -- -- -- -- -- 
70: -- -- -- -- -- -- -- --                         
```

If `68` is shown, the MPU6050 is connected and communicating properly.

---

## Starting the Pi Hub

```bash
python3 pi/pi_hub.py
```

The hub exposes:
- `http://<PI_HOST>:5000/video` - MJPEG video feed
- `http://<PI_HOST>:5000/sensors` - Ultrasonic distance (1.5s validity)
- `http://<PI_HOST>:5000/distance` - Ultrasonic distance (500ms validity)
- `http://<PI_HOST>:5000/motion` - 100 Hz MPU6050 motion readings and fall/impact events
- `http://<PI_HOST>:5000/health` - Hub status
