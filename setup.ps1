# VisionMate Setup Script
# Run this to install all dependencies and verify setup

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "VISIONMATE AI BACKEND SETUP" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "  $pythonVersion" -ForegroundColor Green

if ($pythonVersion -notmatch "Python 3\.(10|11|12)") {
    Write-Host "  Warning: Python 3.10+ recommended" -ForegroundColor Red
}

Write-Host ""

# Install dependencies
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Yellow
Write-Host "  This may take 5-10 minutes on first run..." -ForegroundColor Gray
Write-Host ""

pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "  Dependencies installed successfully!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "  Installation had some errors. Check output above." -ForegroundColor Red
    exit 1
}

Write-Host ""

# Verify database
Write-Host "Verifying database module..." -ForegroundColor Yellow
python database.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "  Database module OK!" -ForegroundColor Green
} else {
    Write-Host "  Database module failed" -ForegroundColor Red
}

Write-Host ""

# Verify object detector
Write-Host "Verifying object detector (will download YOLOv8 if needed)..." -ForegroundColor Yellow
python object_detector.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "  Object detector OK!" -ForegroundColor Green
} else {
    Write-Host "  Object detector failed" -ForegroundColor Red
}

Write-Host ""

# Check webcam
Write-Host "Checking webcam availability..." -ForegroundColor Yellow
$webcamCheck = @"
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print('WEBCAM_OK')
    cap.release()
else:
    print('WEBCAM_FAIL')
"@

$result = python -c $webcamCheck 2>&1

if ($result -match "WEBCAM_OK") {
    Write-Host "  Webcam detected!" -ForegroundColor Green
} else {
    Write-Host "  No webcam detected (tests will use static images)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "SETUP COMPLETE!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Run a quick demo:  python demo.py" -ForegroundColor White
Write-Host "  2. Run full test:     python test\test_integration.py" -ForegroundColor White
Write-Host "  3. Start API server:  python server.py" -ForegroundColor White
Write-Host ""
Write-Host "See README.md for full documentation!" -ForegroundColor Cyan
Write-Host ""
