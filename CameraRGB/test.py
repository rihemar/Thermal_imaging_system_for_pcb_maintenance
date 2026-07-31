import cv2
import numpy as np
from Array_proccessing import *
# from Thermal_imaging_system_for_pcb_maintenance.CameraRGB.old.pcb_extraction import *
from Projection import *
from color_distance import *
import time
from tools import *
# url = "http://192.168.1.15:81/stream"

# cap = cv2.VideoCapture(url)
# if not cap.isOpened():
#     print("Could not connect to IP Webcam")
#     exit()


if __name__ == "__main__":
    while True:
        time.sleep(0.1)  # Add a small delay to reduce CPU usage
        out = display_image()
        # colored_final, output = Array_processing(True)
        # cv2.imshow("Colored Final", colored_final)
        cv2.imshow("Output", out)
        key = cv2.waitKey(1)   # IMPORTANT
        if key == 27:
            break
