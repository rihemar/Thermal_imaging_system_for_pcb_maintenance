import sys
from test import adaptive_saturation
import numpy as np
import cv2

url = "http://192.168.1.181:81/stream"

cap = cv2.VideoCapture(url)
if not cap.isOpened():
    print("Could not connect to IP Webcam")
    exit()


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def compute_size(rect):
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    return maxWidth, maxHeight


def warp_perspective(frame, box):
    rect = order_points(box)

    w, h = compute_size(rect)

    dst = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(frame, M, (w, h))

    return warped


def resize_frame(frame, size):
    h, w = frame.shape[:2]
    new_w = size
    scale = new_w / w
    new_h = int(h * scale)
    return (new_w, new_h)


def boost_saturation(hsv, target_sat=150, value_thresh=30):
    """
    Consistent, shadow-aware saturation boost used everywhere in this file.
    Heavily desaturated feeds need a higher target than a mild boost provides,
    otherwise pixels get amplified but still fall short of the inRange
    saturation floor and get dropped from the mask.
    """
    sat = hsv[:, :, 1]
    value = hsv[:, :, 2]
    shadow_mask = value > value_thresh  # ignore near-black shadow pixels

    if np.any(shadow_mask):
        current_sat = np.mean(sat[shadow_mask])
        if current_sat > 0:
            factor = target_sat / current_sat
            factor = np.clip(factor, 0.5, 3.0)
            hsv[:, :, 1] = np.clip(
                hsv[:, :, 1].astype(np.float32) * factor, 0, 255
            ).astype(np.uint8)

    return hsv


def green_pcb_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = boost_saturation(hsv, target_sat=150)

    lower = (40, 50, 50)
    upper = (90, 255, 255)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((11, 11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def pcb_extraction():
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ret, frame = cap.read()
    if not ret:
        return

    mask = green_pcb_mask(frame)
    output = frame.copy()
    output[mask > 0] = (0, 0, 255)

    # building the smallest rectangle containing the pcb
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        return
    points = coords[:, ::-1].astype(np.float32)
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    # refactoring : flattening (in perspective) the pcb rectangle
    rect = order_points(box)
    for i in range(4):
        output = cv2.circle(output, [int(rect[i][0]), int(rect[i][1])], 1, (255, 0, 0), 3)
        output = cv2.putText(output, str(i), (int(rect[i][0]) + 5, int(rect[i][1]) + 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    warped = warp_perspective(frame, box)

    warped = cv2.resize(warped, resize_frame(warped, 760))
    return warped


def test_pcb_extraction():
    while True:
        warped = pcb_extraction()
        if warped is None:
            print("No PCB detected. Please adjust the camera or the PCB position.")
            continue
        cv2.imshow("Warped PCB", warped)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break


def test_contour_detection():
    while True:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = cap.read()
        if not ret:
            return

        mask = green_pcb_mask(frame)

        # FIX: draw on the original BGR frame, not a mangled re-conversion of HSV
        output = frame.copy()
        output[mask > 0] = (0, 0, 255)

        coords = np.column_stack(np.where(mask > 0))
        if len(coords) == 0:
            continue
        points = coords[:, ::-1].astype(np.float32)
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(output, [box], 0, (0, 0, 255), 2)
        rect = order_points(box)
        for i in range(4):
            output = cv2.circle(output, [int(rect[i][0]), int(rect[i][1])], 1, (255, 0, 0), 3)
            output = cv2.putText(output, str(i), (int(rect[i][0]) + 5, int(rect[i][1]) + 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Contour Detection", output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break


def test_both():
    while True:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = cap.read()
        if not ret:
            return

        mask = green_pcb_mask(frame)
        output = frame.copy()
        output[mask > 0] = (0, 0, 255)

        coords = np.column_stack(np.where(mask > 0))
        if len(coords) == 0:
            return
        points = coords[:, ::-1].astype(np.float32)
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(output, [box], 0, (0, 0, 255), 2)
        rect = order_points(box)
        for i in range(4):
            output = cv2.circle(output, [int(rect[i][0]), int(rect[i][1])], 1, (255, 0, 0), 3)
            output = cv2.putText(output, str(i), (int(rect[i][0]) + 5, int(rect[i][1]) + 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Contour Detection", output)

        warped = warp_perspective(frame, box)
        warped = cv2.resize(warped, resize_frame(warped, 760))
        cv2.imshow("Warped PCB", warped)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break


if __name__ == "__main__":
    argument = sys.argv[1]
    if argument == "warped":
        test_pcb_extraction()
    elif argument == "contour":
        test_contour_detection()
    elif argument == "both":
        test_both()
    else:
        print("Invalid argument. Use 'warped', 'contour', or 'both'.")
