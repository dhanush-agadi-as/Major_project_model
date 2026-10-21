import sys
import os
import cv2
from pathlib import Path

root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Mock VideoCapture so automated tests run headlessly without physical webcam
class MockVideoCapture:
    def __init__(self, *args, **kwargs):
        pass
    def isOpened(self):
        return False
    def read(self):
        return False, None
    def grab(self):
        return False
    def retrieve(self):
        return False, None
    def release(self):
        pass
    def set(self, *args, **kwargs):
        return True
    def get(self, *args, **kwargs):
        return 0

cv2.VideoCapture = MockVideoCapture

# Mock GUI display and blocking waitKey during automated testing
cv2.imshow = lambda *args, **kwargs: None
cv2.waitKey = lambda *args, **kwargs: 0
cv2.destroyAllWindows = lambda *args, **kwargs: None
