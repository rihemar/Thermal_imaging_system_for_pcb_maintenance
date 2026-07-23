import cv2
import numpy as np


def move_edge(box, edge, pixels):
    """
    box : (4,2) float32 clockwise
    edge: 0,1,2,3
    pixels: inward movement
    """

    box = box.astype(np.float32).copy()

    p1 = box[edge]
    p2 = box[(edge + 1) % 4]

    edge_vec = p2 - p1
    edge_vec /= np.linalg.norm(edge_vec)

    # inward normal (clockwise polygon)
    normal = np.array([-edge_vec[1], edge_vec[0]])

    box[edge] += normal * pixels
    box[(edge + 1) % 4] += normal * pixels

    return box

def score_move(binary, old_box, new_box):

    old_mask = np.zeros(binary.shape, np.uint8)
    new_mask = np.zeros(binary.shape, np.uint8)

    cv2.fillPoly(old_mask, [old_box.astype(np.int32)], 255)
    cv2.fillPoly(new_mask, [new_box.astype(np.int32)], 255)

    removed = cv2.bitwise_and(old_mask, cv2.bitwise_not(new_mask))

    removed_pixels = binary[removed > 0]

    ones = np.count_nonzero(removed_pixels)
    zeros = removed_pixels.size - ones

    if removed_pixels.size == 0:
        return 0

    return zeros / removed_pixels.size



def optimize_box(binary, box, max_iterations=10, move_pixels=2, threshold=0.95):
    for edge in range(max_iterations):
        candidate = move_edge(box, edge, move_pixels)

    if score_move(binary, box, candidate) > threshold:
        box = candidate

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



def adjust_saturation(image, factor=1.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    hsv[:, :, 1] *= factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)