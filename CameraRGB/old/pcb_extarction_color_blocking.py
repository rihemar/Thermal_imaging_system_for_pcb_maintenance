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

def resize_frame(frame, max_dim):
    w, h = frame.shape[1], frame.shape[0]
    if w > h:
        scale = max_dim / w
    else:
        scale = max_dim / h
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_frame = cv2.resize(frame, (new_w, new_h))
    return resized_frame

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


def find_group_contour(frame,kernel_size):
    # by default keep kernel_size as 25
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 60, 150)

    # Merge all nearby edges together
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    merged = cv2.dilate(edges, kernel, iterations=2)
    merged = cv2.morphologyEx(
        merged,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        merged,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None, merged

    # eliminate far away contours
    centers = []

    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy, c))

    avg_distances = []

    for i, (cx1, cy1, _) in enumerate(centers):

        dists = []

        for j, (cx2, cy2, _) in enumerate(centers):
            if i == j:
                continue

            d = np.hypot(cx1 - cx2, cy1 - cy2)
            dists.append(d)

        avg_distances.append(np.mean(dists))

    mean_dist = np.mean(avg_distances)
    std_dist = np.std(avg_distances)

    filtered = []

    for (cx, cy, contour), d in zip(centers, avg_distances):

        if d < mean_dist + 2 * std_dist:
            filtered.append(contour)
    if (len(filtered) == 0):
        return None, merged
    
    largest = max(filtered, key=cv2.contourArea)

    return largest, merged


def connect_small_components_algorithm():
    while True:
        print("Contour detection mode")
        ret , frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            exit()
        contour, debug = find_group_contour(frame, kernel_size=50)
        if contour is not None:
            display = frame.copy()
            cv2.drawContours(display, [contour], -1, (0,255,0), 3)
            display = resize_frame(display, 600)
            cv2.imshow("Contour", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def connect_small_components_algorithm_crop():
    while True:
        print("Crop mode")
        ret , frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            exit()
        contour, debug = find_group_contour(frame, kernel_size=50)
        if contour is not None:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cropped = warp_perspective(frame, box)
            cropped = resize_frame(cropped, 600)
            cv2.imshow("Cropped", cropped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()





def convex_hull_algorithm():
    while True:
        ret , frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            exit()
        result = convex_hull_board(frame)
        if result is not None:
            hull, box = result
            cv2.drawContours(frame, [hull], -1, (0,255,0), 2)
            # cropped = warp_perspective(frame, box)
            cv2.imshow("Hull", frame)
            # cv2.imshow("Crop", cropped)


def pca_rectangle(contour):
    """
    Compute a rotated rectangle around a contour using PCA.

    Returns:
        box  : (4,2) integer corner points
        angle: rotation angle in degrees
    """

    # Convert contour to Nx2 float array
    pts = contour.reshape(-1, 2).astype(np.float32)

    # PCA
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(
        pts,
        mean=np.empty((0))
    )

    center = mean[0]

    # Principal direction
    axis1 = eigenvectors[0]
    angle = np.arctan2(axis1[1], axis1[0])

    # Rotation matrix (rotate contour so PCB becomes horizontal)
    c = np.cos(-angle)
    s = np.sin(-angle)

    R = np.array([
        [c, -s],
        [s,  c]
    ])

    rotated = (pts - center) @ R.T

    xmin = np.min(rotated[:, 0])
    xmax = np.max(rotated[:, 0])

    ymin = np.min(rotated[:, 1])
    ymax = np.max(rotated[:, 1])

    rect = np.array([
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax]
    ])

    # Rotate rectangle back
    R_inv = R.T

    box = rect @ R_inv + center

    return np.int32(box), np.degrees(angle)


def pca_algorithm():
    while True:
        ret , frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            exit()
        contour, debug = find_group_contour(frame, kernel_size=50)
        if contour is not None:
            display = frame.copy()
            cv2.drawContours(display, [contour], -1, (0,255,0), 3)

            box, angle = pca_rectangle(contour)

            display = frame.copy()

            cv2.drawContours(display, [box], 0, (0,255,0), 3)

            warped = warp_perspective(frame, box)
            display = resize_frame(display, 600)
            warped = resize_frame(warped, 600)
            cv2.imshow("PCA Rectangle", display)
            cv2.imshow("Warp", warped)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "contour":
        # connect_small_components_algorithm()
        pca_algorithm()
        # convex_hull_algorithm()
    elif arg == "crop":
        connect_small_components_algorithm_crop()
    else:
        print("Invalid argument. Use 'contour' or 'crop'.")
        exit()

    
    
# def order_points(pts):
#     rect = np.zeros((4, 2), dtype="float32")

#     s = pts.sum(axis=1)
#     diff = np.diff(pts, axis=1)

#     rect[0] = pts[np.argmin(s)]  # top-left
#     rect[2] = pts[np.argmax(s)]  # bottom-right
#     rect[1] = pts[np.argmin(diff)]  # top-right
#     rect[3] = pts[np.argmax(diff)]  # bottom-left

#     return rect


# def compute_size(rect):
#     (tl, tr, br, bl) = rect

#     widthA = np.linalg.norm(br - bl)
#     widthB = np.linalg.norm(tr - tl)
#     maxWidth = max(int(widthA), int(widthB))

#     heightA = np.linalg.norm(tr - br)
#     heightB = np.linalg.norm(tl - bl)
#     maxHeight = max(int(heightA), int(heightB))

#     return maxWidth, maxHeight


# def warp_perspective(frame, box):
#     rect = order_points(box)

#     w, h = compute_size(rect)

#     dst = np.array([
#         [0, 0],
#         [w - 1, 0],
#         [w - 1, h - 1],
#         [0, h - 1]
#     ], dtype="float32")

#     M = cv2.getPerspectiveTransform(rect, dst)
#     warped = cv2.warpPerspective(frame, M, (w, h))

#     return warped


# def resize_frame(frame, size):
#     h, w = frame.shape[:2]
#     new_w = size
#     scale = new_w / w
#     new_h = int(h * scale)
#     return (new_w, new_h)


# def detect_board_contour(frame, blur_ksize=25, canny_low=30, canny_high=100,
#                           min_area=1000):
#     """
#     Heavy blur wipes out component-level detail (silkscreen, chips, traces)
#     so Canny only responds to the strong outer edge of the board itself.
#     Returns the 4-point box of the board, and the raw edge map for debugging.
#     """
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
#     edges = cv2.Canny(blur, canny_low, canny_high)

#     # close small gaps so the board outline forms one continuous contour
#     kernel = np.ones((7, 7), np.uint8)
#     closed = cv2.dilate(edges, kernel, iterations=2)
#     closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)

#     contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         return None, edges

#     largest = max(contours, key=cv2.contourArea)
#     if cv2.contourArea(largest) < min_area:
#         return None, edges

#     rect = cv2.minAreaRect(largest)
#     box = cv2.boxPoints(rect)
#     box = np.int32(box)

#     return box, edges


# def trace_components(warped, blur_ksize=3, canny_low=40, canny_high=120):
#     """
#     Light blur keeps small-scale edges alive, so Canny traces the outline
#     of every component sitting on the board instead of just the board itself.
#     Returns an overlay (component outlines drawn on the real picture) and
#     the raw edge map.
#     """
#     gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
#     edges = cv2.Canny(blur, canny_low, canny_high)

#     contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#     overlay = warped.copy()
#     cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)

#     return overlay, edges


# def color_blocking(frame):
#     colors ={"red":(0,10), "orange":(11,25), "yellow":(26,34), "green":(35,85), "blue":(86,125), "purple":(126,160), "red2":(161,179)}
#     color_sub = {"red":(0,255,255),"orange":(20,255,255),"yellow":(30,255,255),"green":(60,255,255),"blue":(120,255,255),"purple":(150,255,255),"red2":(170,255,255)}
    
#     hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     output_frame = hsv_frame.copy()
#     output_frame[:, :, 0] = 0  # Set the hue channel to 0 (black) for all pixels

#     for color, (lower, upper) in colors.items():
#         lower_bound = np.array([lower, 1, 1])
#         upper_bound = np.array([upper, 255, 255])
#         mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
#         output_frame[mask > 0] = color_sub[color]
#     return output_frame

# def cam_test():
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             break
        
#         color_blocked_frame = color_blocking(frame)
#         resized_frame = cv2.resize(color_blocked_frame, (400, 600))
#         cv2.imshow("Color Blocked Frame", resized_frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# def cam():
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             break
        
#         cv2.imshow("Color Blocked Frame", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()



# if __name__ == "__main__":
#     argument = sys.argv[1]
#     if argument == "origin":
#         cam()
#     else:
#     # if argument == "warped":
#     # elif argument == "contour":
#     # elif argument == "both":
#     # else:
#     #     print("Invalid argument. Use 'warped', 'contour', or 'both'.")
#         cam_test()