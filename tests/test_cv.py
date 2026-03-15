import cv2
import numpy as np
from src.vision import detect_fire_cv

def test_detect_fire_empty():
    """Test that a completely black image returns False."""
    img = np.zeros((48, 64, 4), dtype=np.uint8)
    img[:, :, 3] = 255  # Full alpha
    result, _ = detect_fire_cv(img)
    assert not result

def test_detect_fire_orange_patch():
    """Test that an image with a visible orange patch returns True."""
    img = np.zeros((48, 64, 4), dtype=np.uint8)
    img[:, :, 3] = 255
    # The PyBullet orange fire is roughly: color = [1.0, 0.3, 0.0, 1.0] i.e. RGB (255, 76, 0)
    # PyBullet returns RGBA
    img[20:30, 30:40] = [255, 76, 0, 255]
    result, center = detect_fire_cv(img)
    assert result is True
    assert center is not None

def test_detect_fire_red_patch():
    """Test that an image with a red patch (like a wall/obstacle) returns False."""
    img = np.zeros((48, 64, 4), dtype=np.uint8)
    img[:, :, 3] = 255
    # Red obstacle
    img[20:30, 30:40] = [200, 20, 20, 255]
    result, _ = detect_fire_cv(img)
    assert not result

def test_detect_fire_noise():
    """Test that a tiny speck of orange (e.g., 2x2 pixels) is ignored as noise."""
    img = np.zeros((48, 64, 4), dtype=np.uint8)
    img[:, :, 3] = 255
    img[10:12, 10:12] = [255, 76, 0, 255]
    result, _ = detect_fire_cv(img)
    assert not result

if __name__ == "__main__":
    test_detect_fire_empty()
    test_detect_fire_orange_patch()
    test_detect_fire_red_patch()
    test_detect_fire_noise()
    print("All CV tests passed!")
