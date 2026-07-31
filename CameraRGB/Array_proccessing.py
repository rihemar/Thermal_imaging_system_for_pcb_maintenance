import numpy as np
import cv2
import sys
from tools import *
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
    arr = np.loadtxt("./data/CameraArray.txt")
    # print(arr.shape)
    if arr is None:
        print("Failed to load array from CameraArray.txt. Please check the file.")
        return None
    return arr

def ConvertArrayToImage(arr):
    # Normalize to 0-255
    arr_norm = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
    arr_norm = arr_norm.astype(np.uint8)

    # Apply color map
    colored = cv2.applyColorMap(arr_norm, cv2.COLORMAP_JET)

    return colored

def display_image():
    arr = LoadArray()
    if arr is None:
        print("Failed to load array from CameraArray.txt. Please check the file.")
        return None
    # print(f"Array shape: {arr.shape}, min: {np.min(arr)}, max: {np.max(arr)}")
    colored = ConvertArrayToImage(arr)
    colored = resize_width(colored, 600)
    return colored

    

def convert_and_save_image():
    arr = LoadArray()
    if arr is None:
        print("Failed to load array from CameraArray.txt. Please check the file.")
        return None
    colored = ConvertArrayToImage(arr)
    colored = resize_width(colored, 600)
    cv2.imwrite("./data/thermal_frame.jpg",colored)

    

def Array_processing(debug_mode=False):
        
    arr = LoadArray()
    if arr is None:
        print("Failed to load array from CameraArray.txt. Please check the file.")
        return None, None
    colored = ConvertArrayToImage(arr)
    
    if debug_mode:
        cv2.imshow("Colored Image", colored)

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
    while(True):
        colored_final, output = Array_processing()
        if colored_final is not None and output is not None:
            cv2.imshow("Colored Final", colored_final)
            cv2.imshow("Edge Output", output)
        else:
            print("Array processing failed. Please check the input data.")
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break


def debug():
    while(True):
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

        colored_final = colored.copy()
        cv2.polylines(colored_final, [box], True, (0, 255, 0), 2)

        cv2.imshow("Colored Final", colored_final)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

# if __name__ == "__main__":
#     argument = sys.argv[1]
#     if argument == "test":
#         test_Array_processing()
#     elif argument == "debug":
#         debug()
#     else:
#         print("Invalid argument. Use 'test' or 'debug'.")

convert_and_save_image()