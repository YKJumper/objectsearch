import cv2
import numpy as np
import time
from collections import deque

# Global parameters
bitBrightSelector = 0.65
bitThresh = 40

def align_images(image1, image2, s=0.2, numOfKeypoints=500):
    """
    Aligns image2 to image1 using downscaled images and FLANN+LSH matching.
    
    Parameters:
        image1 (ndarray): Reference image.
        image2 (ndarray): Image to be aligned.
        s (float): Downscale factor (e.g., 0.5).
        numOfKeypoints (int): Maximum number of keypoints to detect.
    
    Returns:
        aligned_image1 (ndarray): Original image1 (unaltered).
        aligned_image2 (ndarray): Transformed image2 aligned to image1.
        left_top (tuple): Top-left corner of intersection region.
        right_bottom (tuple): Bottom-right corner of intersection region.
    """
    # 1. Downscale both images
    small_image1 = cv2.resize(image1, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
    small_image2 = cv2.resize(image2, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA)

    # 2. Detect ORB keypoints and descriptors
    orb = cv2.ORB_create(nfeatures=numOfKeypoints)
    keypoints1, descriptors1 = orb.detectAndCompute(small_image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(small_image2, None)

    if descriptors1 is None or descriptors2 is None:
        return image1, image2, (0, 0), (0, 0)

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
    ransacReprojThreshold=5.0,  # Looser threshold for speed
    maxIters=1000,              # Fewer iterations
    confidence=0.98,            # Slightly reduced confidence
    refineIters=10)             # Fewer refinement steps

    if M_small is None:
        return image1, image2, (0, 0), (0, 0)

    # 7. Scale the transform matrix to match original image scale
    M = M_small.copy()
    M[:, 2] /= s

    # 8. Apply transformation to original full-sized image2
    aligned_image2 = cv2.warpAffine(image2, M, (image1.shape[1], image1.shape[0]))

    # 9. Compute intersection (optional external function)
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

def highlight_motion(frame1, frame2, m=1, selectionSide=30):
    global bitThresh
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    aligned1, aligned2, top_left, right_bottom = align_images(gray1, gray2)
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
    annotated_frame = frame2.copy()
    for x, y in zip(xs, ys):
        top_left_corner = (x - selectionSide // 2, y - selectionSide // 2)
        bottom_right_corner = (x + selectionSide // 2, y + selectionSide // 2)
        cv2.rectangle(annotated_frame, top_left_corner, bottom_right_corner, (0, 255, 0), 2)

    return annotated_frame

def crop_frame(frame, crop_percentage=70):
    # Crop the central square region
    height, width = frame.shape[:2]
    crop_size_y, crop_size_x = int(height * (crop_percentage / 100.0)), int(width * (crop_percentage / 100.0))
    center_x, center_y = width // 2, height // 2
    x1 = max(center_x - crop_size_x // 2, 0)
    x2 = min(center_x + crop_size_x // 2, width)
    y1 = max(center_y - crop_size_y // 2, 0)
    y2 = min(center_y + crop_size_y // 2, height)

    return frame[y1:y2, x1:x2]

def play_and_detect(videoFile, start_time=0, end_time=None, save_output=False, output_file="output_with_motion.avi"):
    cap = cv2.VideoCapture(videoFile)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_time = total_frames / fps

    if end_time is None or end_time > total_time:
        end_time = total_time

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))
    current_time = start_time

    # Setup video writer
    writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame.")
        return
    prev_frame = crop_frame(frame)

    while cap.isOpened() and current_time <= end_time:
        frame_number = int(current_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            break
        curr_frame = crop_frame(frame)

        annotated_frame = highlight_motion(prev_frame, curr_frame)

        cv2.imshow("Real-time Object Detection", annotated_frame)
        if save_output:
            writer.write(annotated_frame)

        key = cv2.waitKey(int(1000 / fps))
        if key == 27:  # ESC
            break

        prev_frame = curr_frame
        current_time += 5 / fps

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

# Play from 33s to 60s, enable GPU, smoothing, and save output
play_and_detect("FullCars.mp4", start_time=35, end_time=80, save_output=True)

# "orlan.mp4", start_time=11,
# "FullCars.mp4", start_time=35,
# "pidor2.mp4", start_time=0,
# "stableBalcony.mp4", start_time=18,
# "wavedBalcony.mp4", start_time=0,
# "Stadium.mp4", start_time=0,
