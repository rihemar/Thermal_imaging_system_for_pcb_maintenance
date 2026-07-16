import sys
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


def detect_board_contour(frame, blur_ksize=25, canny_low=30, canny_high=100,
                          min_area=1000):
    """
    Heavy blur wipes out component-level detail (silkscreen, chips, traces)
    so Canny only responds to the strong outer edge of the board itself.
    Returns the 4-point box of the board, and the raw edge map for debugging.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)

    # close small gaps so the board outline forms one continuous contour
    kernel = np.ones((7, 7), np.uint8)
    closed = cv2.dilate(edges, kernel, iterations=2)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, edges

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None, edges

    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    return box, edges


def trace_components(warped, blur_ksize=3, canny_low=40, canny_high=120):
    """
    Light blur keeps small-scale edges alive, so Canny traces the outline
    of every component sitting on the board instead of just the board itself.
    Returns an overlay (component outlines drawn on the real picture) and
    the raw edge map.
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    overlay = warped.copy()
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)

    return overlay, edges


def pcb_extraction():
    """
    Returns:
        real_picture      -> perspective-corrected PCB, true proportions, full color
        component_overlay -> real_picture with every component's contour traced on top
        board_edges       -> raw edge map used to find the board (debugging)
    Returns (None, None, board_edges) if no board was found.
    """
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ret, frame = cap.read()
    if not ret:
        return None, None, None

    box, board_edges = detect_board_contour(frame)
    if box is None:
        return None, None, board_edges

    real_picture = warp_perspective(frame, box)
    real_picture = cv2.resize(real_picture, resize_frame(real_picture, 760))

    component_overlay, _ = trace_components(real_picture)

    return real_picture, component_overlay, board_edges


def test_pcb_extraction():
    """arg: 'warped' -> shows only the real, flattened PCB picture."""
    while True:
        real_picture, _, _ = pcb_extraction()
        if real_picture is None:
            print("No PCB detected. Please adjust the camera or the PCB position.")
            continue
        cv2.imshow("Real PCB (flattened)", real_picture)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break


def test_contour_detection():
    """arg: 'contour' -> shows the board-outline detection step (pre-warp)."""
    while True:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = cap.read()
        if not ret:
            return

        box, board_edges = detect_board_contour(frame)
        output = frame.copy()

        if box is not None:
            cv2.drawContours(output, [box], 0, (0, 0, 255), 2)
            rect = order_points(box)
            for i in range(4):
                output = cv2.circle(output, [int(rect[i][0]), int(rect[i][1])], 1, (255, 0, 0), 3)
                output = cv2.putText(output, str(i), (int(rect[i][0]) + 5, int(rect[i][1]) + 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            print("No PCB detected. Please adjust the camera or the PCB position.")

        cv2.imshow("Board Contour Detection", output)
        cv2.imshow("Board Edge Map", board_edges)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break


def test_both():
    """arg: 'both' -> shows the real picture and the component-traced overlay."""
    while True:
        real_picture, component_overlay, board_edges = pcb_extraction()
        if real_picture is None:
            print("No PCB detected. Please adjust the camera or the PCB position.")
            if board_edges is not None:
                cv2.imshow("Board Edge Map", board_edges)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            continue

        cv2.imshow("Real PCB (flattened)", real_picture)
        cv2.imshow("Component Outlines", component_overlay)
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