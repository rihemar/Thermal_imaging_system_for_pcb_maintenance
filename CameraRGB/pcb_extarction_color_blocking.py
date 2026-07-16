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

def find_group_contour(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 60, 150)

    # Merge all nearby edges together
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,25))
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

    largest = max(contours, key=cv2.contourArea)

    return largest, merged

if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "contour":
        while True:
            print("Contour detection mode")
            ret , frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                exit()
            contour, debug = find_group_contour(frame)
            if contour is not None:
                display = frame.copy()
                cv2.drawContours(display, [contour], -1, (0,255,0), 3)
                cv2.imshow("Contour", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    elif arg == "crop":
        while True:
            print("Crop mode")
            ret , frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                exit()
            rect = cv2.minAreaRect(frame)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cropped = warp_perspective(frame, box)
            cv2.imshow("Cropped", cropped)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        print("Invalid argument. Use 'contour' or 'crop'.")
        exit()
    cap.release()
    cv2.destroyAllWindows()
    
    
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