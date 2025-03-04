import cv2
import numpy as np
import time
import threading
from queue import Queue

global bitBrightSelector
global bitThresh
bitThresh = 40  # Initial threshold value
bitBrightSelector = 0.75 # Initial bright selector value

def dist(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

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

def get_frame_at_time(cap, fps, time_sec, crop_percentage=75):
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

def find_keypoints(image, num_points=5):
    """Find the positions of the num_points lightest and darkest objects."""
    gray = image #cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    
    # Find lightest points (bright objects)
    bitLightThresh = int(bitBrightSelector*(maxVal-minVal)+minVal)
    _, light_thresh = cv2.threshold(gray, bitLightThresh, 255, cv2.THRESH_BINARY)
    light_contours, _ = cv2.findContours(light_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # Find darkest points (dark objects)
    bitDarkThresh = int((1-bitBrightSelector)*(maxVal-minVal)+minVal)
    _, dark_thresh = cv2.threshold(gray, bitDarkThresh, 255, cv2.THRESH_BINARY_INV)
    dark_contours, _ = cv2.findContours(dark_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def get_centroids(contours, max_points):
        # Finding the centroids of the contours
        centroids = []
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:max_points]:  # Take largest contours
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroids.append((cX, cY))
        return centroids

    light_points = get_centroids(light_contours, num_points)
    dark_points = get_centroids(dark_contours, num_points)

    return dark_points + light_points  # Combine light and dark points

def align_images(image1, image2, num_keypoints=10):
    """
    Align image2 to image1 using keypoints and return only the central part of the aligned image.

    :param image1: Reference image (NumPy array)
    :param image2: Image to be aligned (NumPy array)
    :param num_keypoints: Number of keypoints to use for alignment (default: 5)
    :param central_square_percentage: Percentage of the total image area to retain in the center (default: 50%)
    :return: Cropped aligned image, cropped reference image, and transformation matrix
    """
    raw_keypoints1 = find_keypoints(image1, num_keypoints)
    raw_keypoints2 = find_keypoints(image2, num_keypoints)
    
    def exclude_centlal_points(keypoints, center, size):
        return [point for point in keypoints if point[0] < center[0]-size or point[0] > center[0]+size or point[1] < center[1]-size or point[1] > center[1]+size]
        #  Crop the central square region
    height, width = image1.shape[:2]
    image1_center = [width // 2, height // 2]
    keypoints1 = exclude_centlal_points(raw_keypoints1, image1_center, 100)
    
    height, width = image2.shape[:2]
    image2_center = [width // 2, height // 2]
    keypoints2 = exclude_centlal_points(raw_keypoints2, image2_center, 100)

    # Select and order effective key points
    def select_effective_keypoints(keypoints1, keypoints2, accuracy=0.05, exclude_cener=True, center_size=100):
        sorted_keypoints1 = sorted(keypoints1, key=lambda x: x[0]+x[1])
        sorted_keypoints2 = sorted(keypoints2, key=lambda x: x[0]+x[1])
        close_criteria = dist(sorted_keypoints1[-1], sorted_keypoints1[0])*accuracy
        effective_keypoints1 = []
        effective_keypoints2 = []
        for i in range(len(sorted_keypoints1)):
            for j in range(len(sorted_keypoints2)):
                if dist(sorted_keypoints1[i], sorted_keypoints2[j]) < close_criteria :
                    effective_keypoints1.append(sorted_keypoints1[i])
                    effective_keypoints2.append(sorted_keypoints2[j])

        # Building the base of the triangle
        i, j = 0, len(effective_keypoints1)-1
        low_adge1, high_adge1 = effective_keypoints1[i], effective_keypoints1[j]
        low_adge2, high_adge2 = effective_keypoints2[i], effective_keypoints2[j]
        base1, base2 = dist(high_adge1, low_adge1), dist(high_adge2, low_adge2)
        
        while abs(base1-base2)/base2 > accuracy:
            i +=1
            low_adge1, high_adge1 = effective_keypoints1[i], effective_keypoints1[j]
            low_adge2, high_adge2 = effective_keypoints2[i], effective_keypoints2[j]
            base1, base2 = dist(high_adge1, low_adge1), dist(high_adge2, low_adge2)
            if abs(base1-base2)/base2 > accuracy:
                j -=1
                low_adge1, high_adge1 = effective_keypoints1[i], effective_keypoints1[j]
                low_adge2, high_adge2 = effective_keypoints2[i], effective_keypoints2[j]
                base1, base2 = dist(high_adge1, low_adge1), dist(high_adge2, low_adge2)

        # Building the third point of the triangle
        for k in range(len(effective_keypoints1)//2, len(effective_keypoints1)-1):
            if k != i and k != j:
                third1, third2 = effective_keypoints1[k], effective_keypoints2[k]
                low_line1, low_line2 = dist(low_adge1, third1), dist(low_adge2, third2)
                high_line1, high_line2 = dist(high_adge1, third1), dist(high_adge2, third2)
                if abs(high_line1-high_line2)/high_line1< accuracy and abs(low_line1-low_line2)/low_line1< accuracy:
                    effective_keypoints1 = [low_adge1, high_adge1, third1]
                    effective_keypoints2 = [low_adge2, high_adge2, third2]
                    return effective_keypoints1, effective_keypoints2
                
        effective_keypoints1 = [low_adge1, high_adge1]
        effective_keypoints2 = [low_adge2, high_adge2]
        return effective_keypoints1, effective_keypoints2

    true_keypoints1, true_keypoints2 = select_effective_keypoints(keypoints1, keypoints2)

    if len(true_keypoints1) < 3 or len(true_keypoints2) < 3:
        print("Not enough keypoints found for alignment.")
        #return image2, image1, np.eye(2, 3, dtype=np.float32)  # Identity transformation if keypoints are insufficient
        return image2

    # Convert to numpy arrays 
    keypoints1 = np.array(true_keypoints1, dtype=np.float32)  # Need at least 3 points for affine transform
    keypoints2 = np.array(true_keypoints2, dtype=np.float32)

    # Original code
    # Compute affine transformation matrix
    M = cv2.getAffineTransform(keypoints2, keypoints1)
    # Warp image2 to align with image1
    aligned_image = cv2.warpAffine(image2, M, (image1.shape[1], image1.shape[0]))

    return aligned_image

def process_frames(videoFile, startTime, timeStep, timeDelta, frame_queue, endTime=None, sizeThresh=1):
    global bitBrightSelector
    global bitThresh
    cap = cv2.VideoCapture(videoFile)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_time = total_frames / fps
    if endTime is not None:
        endTime = min(endTime, total_time)
    else:
        endTime = total_time
    
    T1 = startTime
    while T1 + timeStep <= endTime:
        raw_image1 = get_frame_at_time(cap, fps, T1)
        raw_image2 = get_frame_at_time(cap, fps, T1 + timeDelta)
        
        # Ensure image1 and image2 are in the correct format
        if raw_image1 is None or raw_image2 is None:
            print("Error: Could not retrieve frames from the video.")
            break
        
        gray1 = cv2.cvtColor(raw_image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(raw_image2, cv2.COLOR_BGR2GRAY)
        
        # Align the two frames
        alligned_image = align_images( gray1, gray2)
        diff = cv2.absdiff(alligned_image, gray1)
        frame_queue.put(diff)
        
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(diff)

        bitThresh = int(bitBrightSelector*(maxVal-minVal)+minVal)
        _, thresh = cv2.threshold(diff, bitThresh, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxScale = 3
        for contour in contours:
            if cv2.contourArea(contour) > sizeThresh:
                x, y, w, h = cv2.boundingRect(contour)
                wext = int(w*(boxScale-1)/2)
                hext = int(h*(boxScale-1)/2)
                cv2.rectangle(raw_image1, (x-wext, y-hext), (x + w + wext, y + h + hext), (0, 0, 255), 2)
        
        frame_queue.put(raw_image1)
        T1 += timeStep
    
    cap.release()
    frame_queue.put(None)  # Signal processing completion

def display_frames(frame_queue, DisplayTime=0.5):
    global bitBrightSelector
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        resize_and_display(frame, title="Detected Difference. bitThresh="+str(bitThresh), delay=DisplayTime)

def detect_changes(videoFile, startTime, timeStep, timeDelta, endTime=None, displayTime=0.5, sizeThresh=1):
    frame_queue = Queue(maxsize=10)
    processing_thread = threading.Thread(target=process_frames, args=(videoFile, startTime, timeStep, timeDelta, frame_queue, endTime, sizeThresh))
    processing_thread.start()
    display_frames(frame_queue, displayTime)
    processing_thread.join()

bitBrightSelector = 0.75
detect_changes("orlan 3.mp4", startTime=25, timeStep=0.3, timeDelta=0.15, endTime=999, displayTime=2.0, sizeThresh=1)