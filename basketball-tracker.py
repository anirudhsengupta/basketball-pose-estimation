import cv2
import numpy as np

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_orange = np.array([4, 100, 80])
    upper_orange = np.array([25, 255, 255])

    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        if area > 800:  # ignore small false positives
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            circularity = 4 * np.pi * area / (cv2.arcLength(cnt, True)**2)
            if 0.4 < circularity < 1.4:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 3)
            else:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 165, 255), 2)

    cv2.imshow("Basketball Tracker", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
