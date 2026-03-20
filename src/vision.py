import cv2
import numpy as np

def detect_fire_cv(rgba_image):
    """Detect orange fire in an RGBA panorama. Returns (True, (cx, cy)) or (False, None)."""
    if rgba_image.dtype != np.uint8:
        rgba_image = np.clip(rgba_image, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Orange fire: RGB (255, 76, 0) -> HSV hue ~2-25
    lower_orange = np.array([2, 100, 100])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 15:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return True, (cx, cy)
    return False, None
