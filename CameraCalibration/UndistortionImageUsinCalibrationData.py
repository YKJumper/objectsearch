import cv2
import numpy as np
import glob

# Load calibration data
calibration = np.load("camera_calibration.npz")
cameraMatrix = calibration["cameraMatrix"]
distCoeffs = calibration["distCoeffs"]

# Load images
images = glob.glob('CameraCalibration/TestImag*.jpg')

for fname in images:
    img = cv2.imread(fname)
    h, w = img.shape[:2]

    # Refine camera matrix
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w, h), 1, (w, h))

    # Undistort image
    dst = cv2.undistort(img, cameraMatrix, distCoeffs, None, newCameraMatrix)

    # Crop and show result
    x, y, w, h = roi
    img = img[y:y+h, x:x+w]
    dst = dst[y:y+h, x:x+w]

    diff = cv2.absdiff(img, dst)
    cv2.imwrite(str(fname)+'-diff.jpg', diff)
