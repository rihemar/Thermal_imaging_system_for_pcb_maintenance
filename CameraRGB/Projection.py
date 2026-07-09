import cv2
import numpy as np

alpha = 0.6  # opacity of img1
beta = 1 - alpha

def getFromFile():
    RGB = cv2.imread("flattenedPCB.jpg")
    Thermal = cv2.imread("thermalHeatMap.jpg")
    RGB = cv2.resize(RGB, (Thermal.shape[1], Thermal.shape[0]))
    return RGB, Thermal

def Projection(RGB, Thermal):
    # RGB, Thermal = getFromFile()
    blended = cv2.addWeighted(RGB, alpha, Thermal, beta, 0)
    return blended



# cv2.imshow("Blended", blended)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

