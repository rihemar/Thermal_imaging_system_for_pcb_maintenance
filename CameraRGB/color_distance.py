

#### MAIN SCRIPT FOR PCB DETECTION USING COLOR DISTANCE METHOD ####


import cv2
import numpy as np

from tools import *
url = "http://192.168.1.19:81/stream"

cap = cv2.VideoCapture(url)
if not cap.isOpened():
    print("Could not connect to IP Webcam")
    exit()

def extract_pcb_mask(frame, border_size=30, threshold=25):

    # Convert to LAB (better color distance)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)

    h, w = lab.shape[:2]

    # Collect pixels from image borders
    top = lab[:border_size, :, :]
    bottom = lab[h-border_size:, :, :]
    left = lab[:, :border_size, :]
    right = lab[:, w-border_size:, :]

    border_pixels = np.concatenate([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3)
    ])

    # Estimate background color
    background = np.mean(border_pixels, axis=0)

    # Compute Euclidean distance from background color
    distance = np.linalg.norm(lab - background, axis=2)

    # Normalize for visualization
    distance_vis = cv2.normalize(
        distance,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Threshold
    _, mask = cv2.threshold(
        distance_vis,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    # Morphological cleanup
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (11, 11)
    )

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask, distance_vis


if __name__ == "__main__":
        
    while True:
        ret , frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        # frame = adjust_saturation(frame, factor=3.0)

        frame = resize_frame(frame, 600)

        mask, distance = extract_pcb_mask(frame,threshold=70)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:

            largest = max(contours, key=cv2.contourArea)

            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            cv2.drawContours(frame, [box], 0, (0,255,0), 2)

            cropped = warp_perspective(frame, box)

            cv2.imshow("Mask", mask)
            cv2.imshow("Distance", distance)
            cv2.imshow("Contour", frame)
            cv2.imshow("Cropped", cropped)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def color_distance_iteration(debug=False):
    ret , frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        return 
    frame = adjust_saturation(frame, factor=1.5)

    frame = resize_frame(frame, 1000)

    mask, distance = extract_pcb_mask(frame,threshold=70)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cropped = None
    if contours:

        largest = max(contours, key=cv2.contourArea)

        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        box = optimize_box(mask, box)
        cv2.drawContours(frame, [box], 0, (0,255,0), 2)

        cropped = warp_perspective(frame, box)
        if(debug):
            cv2.imshow("Mask", mask)
            cv2.imshow("Distance", distance)
            cv2.imshow("Contour", frame)
            cv2.imshow("Cropped", cropped)
    return cropped

