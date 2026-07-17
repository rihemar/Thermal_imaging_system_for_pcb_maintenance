import cv2
import numpy as np
from Array_proccessing import *
# from Thermal_imaging_system_for_pcb_maintenance.CameraRGB.old.pcb_extraction import *
from Projection import *
from color_distance import *



if __name__ == "__main__":
    while True:
        warped = color_distance_iteration()
        if warped is None:
            print("No PCB detected. Please adjust the camera or the PCB position.")
            continue
        cv2.imshow("Warped PCB", warped)
        colored_final, output = Array_processing()
        if colored_final is None or output is None:
            print("Failed to process array. Please check the input data.")
            continue
        cv2.imshow("Colored Final", colored_final)
        cv2.imshow("Edge Output", output)
        blended = Projection(warped, colored_final)
        print("Blended image created successfully.")
        cv2.imshow("Blended", blended)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break