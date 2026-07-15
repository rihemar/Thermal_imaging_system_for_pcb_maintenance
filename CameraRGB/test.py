import cv2
import numpy as np

import cv2
import numpy as np

def adaptive_saturation(img, target_sat):
    """
    target_sat: desired average saturation (0-255)
    """

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Current average saturation (ignore very dark pixels)
    sat = hsv[:, :, 1]
    value = hsv[:, :, 2]

    mask = value > 30  # ignore shadows
    current_sat = np.mean(sat[mask])

    if current_sat == 0:
        return img

    # Calculate adaptive multiplier
    factor = target_sat / current_sat

    # Limit amplification
    factor = np.clip(factor, 0.5, 3.0)

    # Apply
    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1].astype(np.float32) * factor,
        0,
        255
    ).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def boost_saturation(img, strength=2.0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Extract saturation
    s = hsv[:,:,1].astype(np.float32)

    # Stretch saturation range instead of multiplying
    low = np.percentile(s, 5)
    high = np.percentile(s, 95)

    s = (s - low) / (high - low) * 255
    s = np.clip(s, 0, 255)

    # Additional boost
    s = s * strength
    s = np.clip(s, 0, 255)

    hsv[:,:,1] = s.astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

if __name__ == "__main__":

    url = "http://192.168.1.181:81/stream"
    cap = cv2.VideoCapture(url)
    while (True):
        ret , frame = cap.read()
        result = boost_saturation(frame, strength=2.0)
        result = adaptive_saturation(frame,70)
        cv2.imshow("boosted", result)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

