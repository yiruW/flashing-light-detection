import numpy as np
import cv2
import math

# Load Video
cap = cv2.VideoCapture('./test5.mp4')
object_detector = cv2.createBackgroundSubtractorMOG2()

while (cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        frame = cv2.resize(frame,(800,480))
        # hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v = hsv[:,:,2]
        _, t = cv2.threshold(v, 254, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(t, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Remove Bad Blob
        badBlob = []
        for countCnt, cnt in enumerate(contours, start=0):
            # Removing blob smaller than specific sizes
            if cv2.contourArea(cnt) < 50:
                badBlob.append(countCnt)
        contours = np.delete(contours, badBlob)
        
        # Group blobs
        t = cv2.morphologyEx(t, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20)));
        contours, hierarchy = cv2.findContours(t, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # # Remove Bad Blob
        # badBlob = []
        # for countCnt, cnt in enumerate(contours, start=0):
        #     # Removing blob smaller than specific sizes
        #     if cv2.contourArea(cnt) < 100:
        #         badBlob.append(countCnt)
        # contours = np.delete(contours, badBlob)

        # # color filter
        # lower_blue = np.array([60, 35, 140])
        # upper_blue = np.array([180, 255, 255])
        # lower_red = np.array([160,100,50])
        # upper_red = np.array([180,255,255])
        # mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # frame = cv2.bitwise_and(frame, frame, mask = cv2.bitwise_not(mask))
        # mask = cv2.inRange(hsv, lower_red, upper_red)
        # frame = cv2.bitwise_and(frame, frame, mask = cv.bitwise_not(mask))

        # track object by center
        for countCnt, cnt in enumerate(contours, start=0):
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            area = cv2.contourArea(cnt)
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
            cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(frame, str(area), (cX - 20, cY - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('Output', frame)

    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    if cv2.waitKey() & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()