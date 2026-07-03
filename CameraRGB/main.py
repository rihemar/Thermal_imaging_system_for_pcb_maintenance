import cv2

# 0 = first camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break
	
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
#    blur = cv2.GaussianBlur(gray , (5,5),0)
    blur = gray
    edge = cv2.Canny(blur,100,200)
    cv2.imshow("Camera", frame)
    cv2.imshow("Edge",edge)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
