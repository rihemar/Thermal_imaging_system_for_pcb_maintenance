import numpy as np
import cv2

#resizing function
def resizeWidth(frame,size,interpole):
	h,w = frame.shape[:2]
	new_h = int((h*size)/w)
	if(interpole):
		return cv2.resize(frame,(size,new_h),interpolation=cv2.INTER_LANCZOS4)
	else:
		return cv2.resize(frame,(size,new_h),interpolation=cv2.INTER_NEAREST)

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


        if coverage < threshold:
            #print("broke")
            break

        best_rect = current

        #print("found better")

        w -= 2 * shrink
        h -= 2 * shrink

    return best_rect

def LoadArray():
    arr = np.loadtxt("CameraArray.txt")
    return arr

def ConvertArrayToImage(arr):
    # Normalize to 0-255
    arr_norm = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
    arr_norm = arr_norm.astype(np.uint8)

    # Apply color map
    colored = cv2.applyColorMap(arr_norm, cv2.COLORMAP_JET)

    return colored



def Array_processing():
        
    arr = LoadArray()
    if arr is None:
        print("Failed to load array from CameraArray.txt. Please check the file.")
        return None, None
    colored = ConvertArrayToImage(arr)


    edge = cv2.Canny(colored,200,300)

    coords = np.column_stack(np.where(edge > 0))
    if len(coords) != 0:
        points = coords[:, ::-1].astype(np.float32)
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.int32(box)


    rect = cv2.minAreaRect(points)

    best_rect = reduceContour(edge, rect)

    box = cv2.boxPoints(best_rect)
    box = np.int32(box)

    output = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

    cropped = output[box[0][1]:box[2][1],box[0][0]+1:box[1][0]+1]
    if cropped.size == 0:
        print("Cropped image is empty. Please check the PCB detection.")
        return None, None
    cropped = resizeWidth(cropped,760,False)
    # cv2.imshow("cropped",cropped)

    colored_final = colored[box[0][1]:box[2][1],box[0][0]+1:box[1][0]+1]
    colored_final = resizeWidth(colored_final,760,True)
    # cv2.imshow("colored finalll",colored_final)

    # cv2.imwrite("thermalHeatMap.jpg",colored_final)

    # colored = resizeWidth(colored, 760 , True)
    output = resizeWidth(output , 760 , True)
    return colored_final , output


def test_Array_processing():
    colored_final, output = Array_processing()
    if colored_final is not None and output is not None:
        cv2.imshow("Colored Final", colored_final)
        cv2.imshow("Edge Output", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Array processing failed. Please check the input data.")