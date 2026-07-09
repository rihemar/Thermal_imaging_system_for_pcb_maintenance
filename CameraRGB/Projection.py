import cv2
import numpy as np


RGB = cv2.imread("flattenedPCB.jpg")
Thermal = cv2.imread("thermalHeatMap.jpg")

# Make sure both images are the same size
RGB = cv2.resize(RGB, (Thermal.shape[1], Thermal.shape[0]))

alpha = 0.6  # opacity of img1
beta = 1 - alpha

blended = cv2.addWeighted(RGB, alpha, Thermal, beta, 0)

cv2.imshow("Blended", blended)
cv2.waitKey(0)
cv2.destroyAllWindows()

