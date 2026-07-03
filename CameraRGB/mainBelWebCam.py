import cv2

url = "http://192.168.1.144:8080/video"

#cv2.namedWindow("Canny", cv2.WINDOW_NORMAL)
#cv2.resizeWindow("Canny", 1280, 720)
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Could not connect to IP Webcam")
    exit()

granularity = 1

while True:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ret, frame = cap.read()
    if not ret:
        break

#filtering ( raw -> gray -> blur -> canny ) 

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray , (9,9) , 0)
    match granularity:    
       case 1: 
            print("level 1 granularity")
            edge = cv2.Canny(blur,200,300) 
       case 2: 
            print("level 2 granularity")
            edge = cv2.Canny(blur,150,250) 
       case 3: 
            print("level 3 granularity")
            edge = cv2.Canny(blur,100,200) 
       case 4: 
            print("level 4 granularity")
            edge = cv2.Canny(blur,50,150) 

#coutour detection and extraction of pcb frame 
    contour_frame = frame

    contours, _ = cv2.findContours(edge , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    largest = None
    largest_area = 0
    for cntr in contours:
        epsilon = 0.02 * cv2.arcLength(cntr ,True)
        approx = cv2.approxPolyDP(cntr , epsilon , True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > largest_area :
                largest_area  = area
                largest = approx
            cv2.drawContours(contour_frame,[approx], -1, (0, 0, 255), 3)
    if (largest is not None):
        cv2.drawContours(contour_frame, [largest], -1, (0, 0, 255), 6)

# resizing for output
    blur = cv2.resize(blur, (760, 600))
    edge = cv2.resize(edge, (760, 600))
    contour_frame = cv2.resize(contour_frame , (760,600))

# final output frames
 
#    cv2.imshow("BLur",blur)
    cv2.imshow("Canny", edge)
    cv2.imshow("contour_frame",contour_frame)
#event interception (keyboard)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('1'):
        granularity = 1
    elif key == ord('2'):
        granularity = 2
    elif key == ord('3'):
        granularity = 3
    elif key == ord('4'):
        granularity = 4
    elif key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
