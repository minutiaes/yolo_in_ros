import cv2
import numpy as np
import time

corner_h= 6
corner_v = 4
square_size = 40

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((corner_h * corner_v, 3), np.float32)
objp[:, :2] = np.mgrid[0:corner_h, 0:corner_v].T.reshape(-1, 2) * square_size
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # depends on fourcc available camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

time_done = 0
while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (corner_h, corner_v), None)

    if ret:
        #objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        #imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(frame, (corner_h, corner_v), corners2, ret)

        time_current = time.time()
        time_left = 3 - (time_current - time_done)
        time_left = "{:.2f}".format(time_left)
        labelSize, baseLine = cv2.getTextSize(time_left, cv2.FONT_HERSHEY_SIMPLEX, 2, 5)
        cv2.putText(frame, time_left, (baseLine, labelSize[1]+baseLine), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)

        if time_current - time_done > 3:
            objpoints.append(objp)
            imgpoints.append(corners)
            time_done = time.time()

    cv2.imshow('img', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None,
                                                           criteria=criteria)
        print("finito")
        print(ret)
        print(mtx)
        print(dist)
        break
