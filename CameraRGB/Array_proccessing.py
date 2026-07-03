import numpy as np
import cv2

#resizing function
def resizeWidth(frame,size,interpole):
	h,w = frame.shape[:2]
	new_h = int((h*size)/w)
	if(interpole):
		return cv2.resize(frame,(size,new_h),interpolation=cv2.INTER_LANCZOS4)
	else:
		return cv2.resize(frame,(size,new_h))

import cv2
import numpy as np


def rectangle_coverage(edge, box):
    """
    edge : binary edge image (0 or 255)
    box  : 4x2 array of rectangle corners

    returns:
        coverage ratio
    """

    mask = np.zeros_like(edge)

    box = np.int32(box)

    for i in range(4):
        cv2.line(mask,
                 tuple(box[i]),
                 tuple(box[(i + 1) % 4]),
                 255,
                 1)

    overlap = cv2.bitwise_and(mask, edge)

    perimeter_pixels = cv2.countNonZero(mask)

    if perimeter_pixels == 0:
        return 0

    overlap_pixels = cv2.countNonZero(overlap)

    return overlap_pixels / perimeter_pixels

def reduceContour(edge, rect, shrink=1, threshold=0.3):

    center, size, angle = rect

    w, h = size

    best_rect = rect

    while w > 5 and h > 5:

        current = (center, (w, h), angle)

        box = cv2.boxPoints(current)

        coverage = rectangle_coverage(edge, box)

        print(f"Coverage = {coverage:.2f}")

        if coverage < threshold:
            #print("broke")
            break

        best_rect = current

        #print("found better")

        w -= 2 * shrink
        h -= 2 * shrink

    return best_rect



# Load array
arr = np.loadtxt("CameraArray.txt")

# Normalize to 0-255
arr_norm = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
arr_norm = arr_norm.astype(np.uint8)

# Apply color map
colored = cv2.applyColorMap(arr_norm, cv2.COLORMAP_JET)

edge = cv2.Canny(colored,200,300)
coords = np.column_stack(np.where(edge > 0))
edge_1 = edge.copy()
colored_1 = colored.copy()
if len(coords) != 0:
	points = coords[:, ::-1].astype(np.float32)
	rect = cv2.minAreaRect(points)
	box = cv2.boxPoints(rect)
	box = np.int32(box)
	cv2.drawContours(edge_1, [box], 0, (0, 0, 255), 1)
	cv2.drawContours(colored_1, [box], 0, (0, 0, 255), 1)
colored_1 = resizeWidth(colored_1,760,True)
edge_1 = resizeWidth(edge_1,760,True)
#cv2.imshow("edge",edge)
#cv2.imshow("pre output", colored_1)


rect = cv2.minAreaRect(points)

best_rect = reduceContour(edge, rect)

box = cv2.boxPoints(best_rect)
box = np.int32(box)

output = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
#cv2.drawContours(output, [box], 0, (0,0,255), 1)
#cv2.drawContours(colored, [box], 0, (0,0,255), 1)

print("bounding box :")
print(box[0][0])
print(box[1][0])
print(box[0][1])
print(box[2][1])
cropped = output[box[0][1]:box[2][1],box[0][0]+1:box[1][0]+1]
cropped = resizeWidth(cropped,760,True)
cv2.imshow("cropped",cropped)

colored_final = colored[box[0][1]:box[2][1],box[0][0]+1:box[1][0]+1]
colored_final = resizeWidth(colored_final,760,False)
cv2.imshow("colored finalll",colored_final)

colored = resizeWidth(colored, 760 , True)
output = resizeWidth(output , 760 , True)

#cv2.imshow("output",output)
#cv2.imshow("final",colored)

'''
cv2.imshow("edge",edge)
contour_frame = output.copy()
kernel = np.ones((11,11), np.uint8)
edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
edge = cv2.morphologyEx(edge, cv2.MORPH_OPEN, kernel)
cv2.imshow("edgemorph",edge)
edge_blur = cv2.GaussianBlur(edge, (13,13),0)
contours, _ = cv2.findContours(edge_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

largest = None
largest_area = 0

for cnt in contours:
	epsilon = 0.02 * cv2.arcLength(cnt, True)
	approx = cv2.approxPolyDP(cnt, epsilon, True)

	if len(approx) == 4:
		area = cv2.contourArea(approx)
		if area > largest_area:
			largest_area = area
			largest = approx:
		cv2.drawContours(contour_frame,[approx], -1, (0, 0, 255), 3)
		print("found one")
if (largest is not None):
	cv2.drawContours(contour_frame, [largest], -1, (0, 0, 255), 6)

cv2.imshow("contour_frame",contour_frame)
'''

cv2.waitKey(0)
cv2.destroyAllWindows()
