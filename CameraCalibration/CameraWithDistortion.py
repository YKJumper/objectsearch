"""Camera matrix (Intrinsic parameters):
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

# Chessboard dimensions (inner corners per row and column)
CHECKERBOARD = (9, 6)

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

# Load images
images = glob.glob('CameraCalibration/Images/*.jpg')

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
        #resize_and_display(img, title='Chessboard Detection'+str(fname), delay=1000)
        print(f"File {fname} has been processed successfully.")
    else:
        print(f"File {fname} can not be processed.")



# Calibrate camera using points gathered
ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

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
