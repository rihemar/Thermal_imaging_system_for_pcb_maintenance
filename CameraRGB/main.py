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
    colored_final, output = Array_processing()
    blended = Projection(warped, colored_final)
    cv2.imshow("Blended", blended)