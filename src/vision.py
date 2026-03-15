import cv2
import numpy as np

def detect_fire_cv(rgba_image):
    """
    Process the incoming RGBA frame from the drone's FPV camera using OpenCV.
    Returns (True, (cx, cy)) if an orange fire is detected in the frame, False otherwise.
    """
    # PyBullet outputs RGBA uint8 usually, but we ensure it's uint8 for OpenCV
    if rgba_image.dtype != np.uint8:
        rgba_image = np.clip(rgba_image, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGR)
    
    # 2. Convert to HSV
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    
    # 3. Define Orange threshold 
    # The PyBullet orange fire is roughly: [1.0, 0.3, 0.0] -> RGB (255, 76, 0)
    # In HSV (H is 0-179 in OpenCV):
    lower_orange = np.array([2, 100, 100])
    upper_orange = np.array([25, 255, 255])
    
    # 4. Create Mask
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # 5. Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 15:  # Balanced threshold for robustness and range
            # Calculate the center of the contour
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return True, (cx, cy)
                
    return False, None
