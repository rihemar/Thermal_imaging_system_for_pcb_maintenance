import cv2
import numpy as np
from tools import *
import sys


url = "http://192.168.1.181:81/stream"

cap = cv2.VideoCapture(url)
if not cap.isOpened():
    print("Could not connect to IP Webcam")
    exit()

while True:
    ret , frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    frame = resize_frame(frame, 400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 100)
    # edges = resize_frame(edges, 760)
    cv2.imshow("Edges", edges)

    # Get coordinates of all edge pixels
    points = np.column_stack(np.where(edges > 0))
    print(f"Number of edge points detected: {len(points)}")
    if len(points) > 0:
        # Convert (row, col) -> (x, y)
        points = points[:, ::-1].astype(np.float32)

        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
        # frame = resize_frame(frame, 760)
        cv2.imshow("contour", frame)
        cropped = warp_perspective(frame, box)
        # cropped = resize_frame(cropped, 760)
        cv2.imshow("Cropped", cropped)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()