"""
Camera matrix (Intrinsic parameters):
 [[1.45744886e+03 0.00000000e+00 9.01478613e+02]
 [0.00000000e+00 1.45594373e+03 4.96952535e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]

Distortion coefficients:
 [[ 0.11957122 -0.38768829  0.00137636  0.00394898  0.3468746 ]]

Calibration data saved successfully!
"""
"""
Camera matrix (Intrinsic parameters):
 [[1.53218978e+03 0.00000000e+00 9.13238112e+02]
 [0.00000000e+00 1.53231008e+03 4.97674328e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]

Distortion coefficients:
 [[ 1.36208940e-01 -3.60849922e-01 -3.06140615e-04  2.00650194e-03  1.37895353e-01]]
Calibration data saved successfully!
"""
"""
Camera matrix (Intrinsic parameters) ./1847/ from CalibrationVideoPlane.mp4 1.35:
 [[1.45744885e+03 0.00000000e+00 9.01478611e+02]
 [0.00000000e+00 1.45594372e+03 4.96952534e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]

Distortion coefficients:
 [[ 1.1957122e-01 -3.8768829e-01  0.00137636  0.00394898  0.34687461]]

Calibration data saved successfully!
"""
"""
Camera matrix (Intrinsic parameters) ./1837/ from CalibrationVideo.mp4 1.35:
 [[1.52907446e+03 0.00000000e+00 8.95326660e+02]
 [0.00000000e+00 1.52834562e+03 5.10921972e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]

Distortion coefficients:
 [[ 0.13334048 -0.33984317  0.00147857 -0.00239344  0.24224616]]

Calibration data saved successfully!
"""
"""
Camera matrix (Intrinsic parameters) ./1818/  from CalibrationVideo.mp4 1.3:
 [[1.53120091e+03 0.00000000e+00 9.09722794e+02]
 [0.00000000e+00 1.52858664e+03 5.15020657e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]

Distortion coefficients:
 [[ 1.50308333e-01 -4.36405056e-01 -2.75359907e-06 -6.48009132e-04 4.26955425e-01]]
"""
"""
Camera matrix (Intrinsic parameters) from video ./1803/  from CalibrationVideoPlane.mp4 1.0:
 [[1.52022685e+03 0.00000000e+00 9.09186642e+02]
 [0.00000000e+00 1.52020797e+03 5.05568593e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]

Distortion coefficients:
 [[ 0.14200387 -0.35543693 -0.00083289  0.00134735  0.2573171 ]]
"""

"""Camera matrix (Intrinsic parameters) form template image:
 [[3.08362261e+03 0.00000000e+00 2.00146955e+03]
 [0.00000000e+00 3.07925977e+03 1.47979316e+03]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]

Distortion coefficients:
 [[ 0.13847544 -0.38674022 -0.00190574  0.00151758  0.14959996]]

Calibration data saved successfully!
"""

import cv2
import numpy as np
import glob
from datetime import datetime
from pathlib import Path

# Chessboard dimensions (inner corners per row and column)
CHECKERBOARD = (6, 9)

def get_frame_at_time(cap, fps, time_sec, crop_percentage=95):
    frame_number = int(time_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        return None
    
    # Crop the central square region
    height, width = frame.shape[:2]
    crop_size_y, crop_size_x = int(height * (crop_percentage / 100.0)), int(width * (crop_percentage / 100.0))
    center_x, center_y = width // 2, height // 2
    x1 = max(center_x - crop_size_x // 2, 0)
    x2 = min(center_x + crop_size_x // 2, width)
    y1 = max(center_y - crop_size_y // 2, 0)
    y2 = min(center_y + crop_size_y // 2, height)

    return frame[y1:y2, x1:x2]

def get_calibration_image_set(videoFile, startTime, timeStep, templateCnt):
    cap = cv2.VideoCapture(videoFile)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    endTime = total_frames / fps
    
    i = 0
    T1 = startTime
    folderName = datetime.now().strftime("%H%M")
    Path(folderName).mkdir(parents=True, exist_ok=True) 
    while T1 + timeStep <= endTime and i < templateCnt:
        image = get_frame_at_time(cap, fps, T1)
        cv2.imwrite(folderName+'/testImage_'+str(i)+'.jpg', image)
        i += 1
        T1 += timeStep

    return folderName


def resize_and_display(image, screen_width=1920, screen_height=1080, title="Detected Difference", delay=0.5):
    """
    Resize the given image while maintaining aspect ratio to fit within the specified screen dimensions,
    then display the resized image for a given delay before automatically closing.

    :param image: Input image (NumPy array)
    :param screen_width: Screen width in pixels (default: 1920 for HD)
    :param screen_height: Screen height in pixels (default: 1080 for HD)
    :param title: Title of the displayed image window (default: "Detected Difference")
    :param delay: Time to display the image in seconds before closing automatically (default: 0.5s)
    """
    height, width = image.shape[:2]
    if width > screen_width or height > screen_height:
        scale = min(screen_width / width, screen_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        cv2.imshow(title, resized_image)
    else:
        cv2.imshow(title, image)
    
    
    cv2.moveWindow(title, 0, 0)
    cv2.waitKey(int(delay * 1000))
    cv2.destroyAllWindows()

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all images.
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Prepare a grid of object points (0,0,0), (1,0,0), ..., (8,5,0)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Create calibaration image set
folderName = get_calibration_image_set('VideoCalibrationPlane.mp4', 0, 1.35, 20)
# Load images
images = glob.glob(folderName+'/testImage_*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)

        # Refine corner locations
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
 
        # Draw corners for visualization (optional)
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        resize_and_display(img, title='Chessboard Detection'+str(fname), delay=1000)
        print(f"File {fname} has been processed successfully.")
    else:
        print(f"File {fname} can not be processed.")



# Calibrate camera using points gathered
ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print calibration results
print("\nCamera matrix (Intrinsic parameters):\n", cameraMatrix)
print("\nDistortion coefficients:\n", distCoeffs)

# Save the results for later use
np.savez("camera_calibration.npz",
         cameraMatrix=cameraMatrix,
         distCoeffs=distCoeffs,
         rvecs=rvecs,
         tvecs=tvecs)

print("\nCalibration data saved successfully!")
