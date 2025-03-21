import cv2
import math
import numpy as np
import time
import threading
from queue import Queue 

global bitBrightSelector
global bitThresh
global rotationSpeed 
global numOfKeypoints
global kpGraphRigidity
kpGraphRigidity = 2
numOfKeypoints = 500
rotationSpeed = 20 # The camera rotation speed in degrees per second
bitThresh = 40  # Initial threshold value
bitBrightSelector = 0.75 # Initial bright selector value

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

def get_frame_at_time(cap, fps, time_sec, crop_percentage=70):
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

def process_frames(videoFile, startTime, timeStep, timeDelta, frame_queue, endTime=None, sizeThresh=1):
    global bitBrightSelector
    global bitThresh
    global rotationSpeed
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
        alligned_image1, alligned_image2, top_left, right_bottom = align_images( gray1, gray2)
        diff = cv2.absdiff(alligned_image1, alligned_image2)
        frame_queue.put(diff)
        
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(diff)

        bitThresh = int(bitBrightSelector*(maxVal-minVal)+minVal)
        # bitThresh = int(bitBrightSelector*maxVal)
        _, thresh = cv2.threshold(diff, bitThresh, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxScale = 3
        for contour in contours:
            if cv2.contourArea(contour) > sizeThresh:
                x, y, w, h = cv2.boundingRect(contour)
                x, y = x + top_left[0], y + top_left[1]
                wext = int(w*(boxScale-1)/2)
                hext = int(h*(boxScale-1)/2)
                cv2.rectangle(raw_image2, (x-wext, y-hext), (x + w + wext, y + h + hext), (0, 0, 255), 2)
        # Hihglight the maxLoc position
        selectionSide = 30
        cv2.rectangle(raw_image2, (maxLoc[0]-selectionSide//2 + top_left[0], maxLoc[1]-selectionSide//2 + top_left[1]), (maxLoc[0] + selectionSide//2 + top_left[0], maxLoc[1] + selectionSide//2 + top_left[1]), (0, 255, 0), 2)
        
        frame_queue.put(raw_image2)
        T1 += timeStep
    
    cap.release()
    frame_queue.put(None)  # Signal processing completion

def align_images(image1, image2):
    global numOfKeypoints
    global kpGraphRigidity
    orb = cv2.ORB_create(numOfKeypoints)
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)[:50]
    
    if len(matches) < 10:
        return image2, image2, (0, 0), (image1.shape[1], image1.shape[0])
    
    src_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Verify keypoints graph rigidity by checking pairwise distances
    rigid_matches = []
    for i in range(len(matches)):
        for j in range(i + 1, len(matches)):
            d_src = np.linalg.norm(src_pts[i] - src_pts[j])
            d_dst = np.linalg.norm(dst_pts[i] - dst_pts[j])
            if abs(d_src - d_dst) <= kpGraphRigidity:  # Threshold for rigid transformation
                rigid_matches.append(matches[i])
                rigid_matches.append(matches[j])
    
    rigid_matches = list(set(rigid_matches))  # Remove duplicates
    if len(rigid_matches) < 10:
        print(f"The keypoints rigid graph contains less than 10 points: {len(rigid_matches)}") 
        return image2, image2, (0, 0), (image1.shape[1], image1.shape[0])
    
    src_pts = np.float32([keypoints2[m.trainIdx].pt for m in rigid_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints1[m.queryIdx].pt for m in rigid_matches]).reshape(-1, 1, 2)
    
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    align_image2 = cv2.warpAffine(image2, M, (image1.shape[1], image1.shape[0]))

    # Set the bounding box of the largest dark region
    align_image1, align_image2, left_top, right_bottom = compute_intersection(image1, align_image2, M)

    return align_image1, align_image2, left_top, right_bottom

def compute_intersection(image1, image2, M):
    """
    Crops image1 and image2 to their intersection region based on transformation matrix M.
    
    Parameters:
        image1 (numpy.ndarray): First aligned image.
        image2 (numpy.ndarray): Second aligned image.
        M (numpy.ndarray): 2x3 Affine transformation matrix.
    
    Returns:
        tuple: Cropped image1 and image2 containing only the intersection region.
    """
    # Validate matrix
    if M is None:
        raise ValueError("Affine transformation matrix M is None")
    
    # Get image dimensions
    h, w = image1.shape[:2]
    
    # Define corners of image1
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
    
    # Transform corners using M
    transformed_corners = cv2.transform(np.array([corners]), M)[0]
    
    # Extract individual transformed corner coordinates
    left_top_x, left_top_y         = transformed_corners[0]
    right_top_x, right_top_y       = transformed_corners[1]
    left_bottom_x, left_bottom_y   = transformed_corners[2]
    right_bottom_x, right_bottom_y = transformed_corners[3]
    
    # Compute bounding box limits using specific corner coordinates
    x_min = max(0, int(np.ceil(max(left_top_x, left_bottom_x))))
    y_min = max(0, int(np.ceil(max(left_top_y, right_top_y))))
    x_max = min(w, int(np.floor(min(right_bottom_x, right_top_x))))
    y_max = min(h, int(np.floor(min(right_bottom_y, left_bottom_y))))
    
    # Ensure valid intersection
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("No valid intersection found.")
    
    # Crop images to the intersection region
    cropped_image1 = image1[y_min:y_max, x_min:x_max]
    cropped_image2 = image2[y_min:y_max, x_min:x_max]
    
    return cropped_image1, cropped_image2, (x_min, y_min), (x_max, y_max)

def display_frames(frame_queue, DisplayTime=0.5):
    global bitBrightSelector
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        resize_and_display(frame, title="Detected Difference. bitThresh="+str(bitThresh), delay=DisplayTime)

def process_video(videoFile, startTime, timeStep, timeDelta, endTime=None, displayTime=0.5, sizeThresh=1):
    frame_queue = Queue(maxsize=10)
    processing_thread = threading.Thread(target=process_frames, args=(videoFile, startTime, timeStep, timeDelta, frame_queue, endTime, sizeThresh))
    processing_thread.start()
    display_frames(frame_queue, displayTime)
    processing_thread.join()

bitBrightSelector = 0.75
process_video("FullCars.mp4", startTime=33, timeStep=0.33, timeDelta=0.15, endTime=999, displayTime=0.33, sizeThresh=1)

# "orlan.mp4", startTime=11,
# "cars.mp4", startTime=33,
# "pidor2.mp4", startTime=0,
# "stableBalcony.mp4", startTime=18,
# "wavedBalcony.mp4", startTime=0,
# "Stadium.mp4", startTime=0,