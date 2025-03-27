import cv2
import numpy as np

# Global parameters
bitBrightSelector = 0.75
bitThresh = 40
kpGraphRigidity = 2
numOfKeypoints = 500

def align_images(image1, image2):
    orb = cv2.ORB_create(numOfKeypoints)
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

    if descriptors1 is None or descriptors2 is None:
        return image1, image2, (0, 0)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)[:50]

    if len(matches) < 10:
        return image1, image2, (0, 0)

    src_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if M is None:
        return image1, image2, (0, 0)

    aligned_image2 = cv2.warpAffine(image2, M, (image1.shape[1], image1.shape[0]))
    # Set the bounding box of the largest dark region
    align_image1, align_image2, left_top, right_bottom = compute_intersection(image1, aligned_image2, M)

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
        x_min, y_min, x_max, y_max = 0, 0, w, h
        # raise ValueError("No valid intersection found.")
    
    # Crop images to the intersection region
    cropped_image1 = image1[y_min:y_max, x_min:x_max]
    cropped_image2 = image2[y_min:y_max, x_min:x_max]
    
    return cropped_image1, cropped_image2, (x_min, y_min), (x_max, y_max)

def highlight_motion(frame1, frame2):
    global bitThresh
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    aligned1, aligned2, top_left, right_bottom = align_images(gray1, gray2)
    diff = cv2.absdiff(aligned1, aligned2)

    minVal, maxVal, _, maxLoc = cv2.minMaxLoc(diff)
    # bitThresh = int(bitBrightSelector * (maxVal - minVal) + minVal)
    # _, thresh = cv2.threshold(diff, bitThresh, 255, cv2.THRESH_BINARY)
    # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated_frame = frame2.copy()
    # for contour in contours:
    #     if cv2.contourArea(contour) > 1:
    #         x, y, w, h = cv2.boundingRect(contour)
    #         x, y = x + top_left[0], y + top_left[1]
    #         cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Highlight max diff location
    selectionSide = 30
    cv2.rectangle(annotated_frame,
                  (maxLoc[0] - selectionSide // 2, maxLoc[1] - selectionSide // 2),
                  (maxLoc[0] + selectionSide // 2, maxLoc[1] + selectionSide // 2),
                  (0, 255, 0), 4)
    return annotated_frame

def play_and_detect(videoFile, start_time=0, end_time=None):
    cap = cv2.VideoCapture(videoFile)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_time = total_frames / fps

    if end_time is None or end_time > total_time:
        end_time = total_time

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))
    current_time = start_time

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame.")
        return

    while cap.isOpened() and current_time <= end_time:
        frame_number = int(current_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, curr_frame = cap.read()
        if not ret:
            break
        start_tick = cv2.getTickCount()  #Start timing
    
        detected_frame = highlight_motion(prev_frame, curr_frame)
    
        end_tick = cv2.getTickCount()  #End timing
        time_ms = (end_tick - start_tick) / cv2.getTickFrequency() * 1000  # convert to ms
    
        cv2.imshow("Real-time Object Detection", detected_frame)
    
        if cv2.waitKey(30) & 0xFF == 27:  # ESC key to stop
            break
    
        prev_frame = curr_frame
        current_time += 5 / fps
        print(f"Real-time Object Detection - {time_ms:.2f} ms")

    cap.release()
    cv2.destroyAllWindows()

# Run real-time detection
play_and_detect("FullCars.mp4", start_time=35, end_time=80)
