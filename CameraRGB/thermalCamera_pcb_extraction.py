import cv2
import numpy as np


def resizeFrame(frame, size):
	w, h = frame.shape[:2]
	new_w = size
	scale = new_w / w
	new_h = int(h * scale)
	return (new_w, new_h)


pcb_raw = cv2.imread("thermalCameraFrame.jpg")

if pcb_raw is None:
	print("Image not found or path is wrong")
	exit()

pcb_raw = cv2.resize(pcb_raw, resizeFrame(pcb_raw, 60))

granularity = 2  # default value

while True:

	hsv = cv2.cvtColor(pcb_raw, cv2.COLOR_BGR2HSV)

	hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 3)
	hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

	enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#	gray = cv2.cvtColor(pcb_raw, cv2.COLOR_BGR2GRAY)
	gray = pcb_raw.copy()
#	blur = cv2.GaussianBlur(gray, (9, 9), 0)
	blur = enhanced.copy()

	b, g, r = cv2.split(pcb_raw)
	
	gx_r = cv2.Sobel(r, cv2.CV_32F, 1, 0)
	gy_r = cv2.Sobel(r, cv2.CV_32F, 0, 1)

	gx_g = cv2.Sobel(g, cv2.CV_32F, 1, 0)
	gy_g = cv2.Sobel(g, cv2.CV_32F, 0, 1)

	gx_b = cv2.Sobel(b, cv2.CV_32F, 1, 0)
	gy_b = cv2.Sobel(b, cv2.CV_32F, 0, 1)
	
	mag_r = cv2.magnitude(gx_r, gy_r)
	mag_g = cv2.magnitude(gx_g, gy_g)
	mag_b = cv2.magnitude(gx_b, gy_b)

	gradient = np.sqrt(mag_r**2 + mag_g**2 + mag_b**2)

	gradient = cv2.normalize(gradient,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)


	gradient = cv2.resize(gradient, resizeFrame(gradient, 760))

	cv2.imshow("Color Gradient", gradient)

# -------------------------
	# GRANULARITY CONTROL
	# -------------------------
	match granularity:
		case 1:
			edge = cv2.Canny(blur, 200, 300)
		case 2:
			edge = cv2.Canny(blur, 150, 250)
		case 3:
			edge = cv2.Canny(blur, 100, 200)
		case 4:
			edge = cv2.Canny(blur, 50, 150)

	contour_frame = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

	contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	largest = None
	largest_area = 0

	for cntr in contours:

		epsilon = 0.02 * cv2.arcLength(cntr, True)
		approx = cv2.approxPolyDP(cntr, epsilon, True)

		if len(approx) == 4:

			area = cv2.contourArea(approx)

			if area > largest_area:
				largest_area = area
				largest = approx

		cv2.drawContours(contour_frame, [approx], -1, (0, 0, 255), 2)

	if largest is not None:
		cv2.drawContours(contour_frame, [largest], -1, (0, 255, 0), 4)
	
	cv2.imshow("blur",blur)
	cv2.imshow("original",pcb_raw)
	cv2.imshow("output", contour_frame)

	# -------------------------
	# KEY CONTROL
	# -------------------------
	key = cv2.waitKey(1) & 0xFF

	if key == ord('1'):
		print("1")
		granularity = 1
	elif key == ord('2'):
		print("2")
		granularity = 2
	elif key == ord('3'):
		print("3")		
		granularity = 3
	elif key == ord('4'):
		print("4")
		granularity = 4
	elif key == ord('q'):
		break

cv2.destroyAllWindows()
