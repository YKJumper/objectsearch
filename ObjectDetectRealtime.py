import cv2
import numpy as np

def align_images(image1, image2, keypoints1, descriptors1, keypoints2, descriptors2, s):
    """
    Aligns image2 to image1 using downscaled images and FLANN+LSH matching.
    
    Parameters:
        image1 (ndarray): Reference image.
        image2 (ndarray): Image to be aligned.
        keypoints1 : Resized reference image keypoints.
        descriptors1 : Resized reference image keypoint descriptors.
        keypoints2 : Resized to be aligned image keypoints.
        descriptors2 : Resized to be aligned image keypoint descriptors.
        s (float): Downscale factor (e.g., 0.5).
        numOfKeypoints (int): Maximum number of keypoints to detect.
    
    Returns:
        aligned_image1 (ndarray): Original image1 (unaltered).
        aligned_image2 (ndarray): Transformed image2 aligned to image1.
        left_top (tuple): Top-left corner of intersection region.
        right_bottom (tuple): Bottom-right corner of intersection region.
    """
    # Get image dimensions
    h, w = image1.shape[:2]

    # 3. Use FLANN + LSH for binary descriptors
    index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # 4. Apply Lowe's ratio test
    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)


    if len(good_matches) < 10:
        return image1, image2, (0, 0), (0, 0)

    # 5. Extract matched points
    src_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 6. Estimate affine transform on small images
    # M_small, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    M_small, inliers = cv2.estimateAffine2D(
    src_pts, dst_pts,
    method=cv2.RANSAC,
    ransacReprojThreshold=5.0)

    if M_small is None:
        return image1, image2, (0, 0), (0, 0)

    # 7. Scale the transform matrix to match original image scale
    M = M_small.copy()
    M[:, 2] /= s

    # 8. Apply transformation to original full-sized image2
    aligned_image2 = cv2.warpAffine(image2, M, (w, h))

    # 9. Compute intersection (optional external function)
    x_min, y_min, x_max, y_max = compute_intersection(h, w, M)
    # Crop images to the intersection region
    aligned_image1 = image1[y_min:y_max, x_min:x_max]
    aligned_image2 = aligned_image2[y_min:y_max, x_min:x_max]

    return aligned_image1, aligned_image2, (x_min, y_min), (x_max, y_max)

def compute_intersection(h, w, M):
    """
    Returns image1 and image2 intersection region coordinates based on transformation matrix M.
    
    Parameters:
        h, w (integer) -- image1 height and width
        M (numpy.ndarray): 2x3 Affine transformation matrix.
    
    Returns:
        tuple: the intersection left upper and right bottom region coordinates.
    """
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
        print("Error: No valid intersection found.")
    
    return x_min, y_min, x_max, y_max

def highlight_motion(gray1, gray2, frame2, keypoints1, descriptors1, keypoints2, descriptors2, s, m=1, selectionSide=30):
    aligned1, aligned2, top_left, right_bottom = align_images(gray1, gray2, keypoints1, descriptors1, keypoints2, descriptors2, s)
    diff = cv2.absdiff(aligned1, aligned2)

    # Flatten the difference image and find indices of top m brightest pixels
    flat = diff.flatten()
    if m >= len(flat):
        m = len(flat)

    top_indices = np.argpartition(flat, -m)[-m:]
    top_indices = top_indices[np.argsort(flat[top_indices])[::-1]]  # sort descending by intensity

    # Convert flat indices to (x, y) coordinates
    h, w = diff.shape
    ys, xs = np.unravel_index(top_indices, (h, w))

    # Annotate the frame
    annotated_frame = frame2
    for x, y in zip(xs, ys):
        x, y = x + top_left[0], y + top_left[1]
        top_left_corner = (x - selectionSide // 2, y - selectionSide // 2)
        bottom_right_corner = (x + selectionSide // 2, y + selectionSide // 2)
        cv2.rectangle(annotated_frame, top_left_corner, bottom_right_corner, (0, 255, 0), 2)

    return annotated_frame

def crop_frame(frame, crop_percentage):
    # Crop the central square region
    height, width = frame.shape[:2]
    crop_size_y, crop_size_x = int(height * (crop_percentage / 100.0)), int(width * (crop_percentage / 100.0))
    center_x, center_y = width // 2, height // 2

    x1 = max(center_x - crop_size_x // 2, 0)
    x2 = min(center_x + crop_size_x // 2, width)
    y1 = max(center_y - crop_size_y // 2, 0)
    y2 = min(center_y + crop_size_y // 2, height)

    return frame[y1:y2, x1:x2]

def play_and_detect(videoFile, start_time, end_time, fpsStep, crop_percentage, s, numOfKeypoints, save_output=False, output_file="output_with_motion.avi"):
    orb = cv2.ORB_create(nfeatures=numOfKeypoints)
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

    prev_frame = crop_frame(prev_frame, crop_percentage)
    # 1. Downscale image
    prev_small = cv2.resize(prev_frame, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
    # 2. Detect ORB keypoints and descriptors
    prev_keypoints, prev_descriptors = orb.detectAndCompute(prev_small, None)
    # 3. Convert frame to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    # 4. Check if descriptors are None or too few keypoints
    if  prev_descriptors is None  or len( prev_descriptors) < 2:
        raise ValueError("Not enoght keypoints found in the first frame.")


    while cap.isOpened() and current_time <= end_time:
        frame_number = int(current_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, curr_frame = cap.read()
        if not ret:
            break
    
        start_tick = cv2.getTickCount()  #Start timing
    
        curr_frame = crop_frame(curr_frame, crop_percentage)
        # 1. Downscale image
        curr_small = cv2.resize(curr_frame, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
        # 2. Detect ORB keypoints and descriptors
        curr_keypoints, curr_descriptors = orb.detectAndCompute(curr_small, None)
        # 3. Convert frame to grayscale
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        # 4. Check if descriptors are None or too few keypoints
        if  curr_descriptors is None  or len(curr_descriptors) < 2:
            print("Not enoght keypoints found in the next frame.")
            detected_frame = prev_frame
        else:
            detected_frame = highlight_motion(prev_gray, curr_gray, curr_frame, prev_keypoints, prev_descriptors, curr_keypoints, curr_descriptors, s)

        prev_frame = curr_frame
        prev_small = curr_small
        prev_gray = curr_gray
        prev_keypoints = curr_keypoints
        prev_descriptors = curr_descriptors

        current_time += fpsStep / fps

        end_tick = cv2.getTickCount()  #End timing
        time_ms = (end_tick - start_tick) / cv2.getTickFrequency() * 1000  # convert to ms

        cv2.imshow("Real-time Object Detection", detected_frame)
    
        if cv2.waitKey(30) & 0xFF == 27:  # ESC key to stop
            break
        print(f"Real-time Object Detection - {time_ms:.2f} ms")

    cap.release()
    cv2.destroyAllWindows()

# Run real-time detection
play_and_detect("FullCars.mp4", start_time=35, end_time=80, fpsStep=3, crop_percentage = 70, s=0.25, numOfKeypoints=500, save_output=False)