import cv2
import numpy as np
from Array_proccessing import *
from pcb_extraction import *
from Projection import *


while True:
    warped = pcb_extraction()
    if warped is None:
        print("No PCB detected. Please adjust the camera or the PCB position.")
        continue
    cv2.imshow("Warped PCB", warped)
    colored_final, output = Array_processing()
    cv2.imshow("Colored Final", colored_final)
    cv2.imshow("Edge Output", output)
    blended = Projection(warped, colored_final)
    print("Blended image created successfully.")
    cv2.imshow("Blended", blended)