import cv2
import numpy as np
import heapq

def align_images(image1, image2, keypoints1, descriptors1, keypoints2, descriptors2, s, matcher):
    """
    Aligns image2 to image1 using FLANN+LSH matching and affine transformation.

    Parameters:
        image1 (ndarray): Reference image.
        image2 (ndarray): Image to align.
        keypoints1, keypoints2: Keypoints from downscaled images.
        descriptors1, descriptors2: Descriptors from downscaled images.
        s (float): Downscale factor.

    Returns:
        aligned_image1 (ndarray): Cropped reference image.
        aligned_image2 (ndarray): Aligned and cropped image2.
        left_top (tuple): Top-left intersection coordinates.
        right_bottom (tuple): Bottom-right intersection coordinates.
    """
    h, w = image1.shape[:2]
    
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    if len(good_matches) < 10:
        return image1, image2, (0, 0), (0, 0)

    # Extract matched coordinates
    src_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estimate affine transformation using RANSAC
    M_small, _ = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)

    if M_small is None:
        return image1, image2, (0, 0), (0, 0)

    # Scale matrix to original image size
    M_small[:, 2] /= s

    # Warp image2 using scaled transform
    aligned_image2 = cv2.warpAffine(image2, M_small, (w, h))

    # Compute intersection region (external function)
    x_min, y_min, x_max, y_max = compute_intersection(h, w, M_small)

    # Crop both images to the overlapping region
    aligned_image1 = image1[y_min:y_max, x_min:x_max]
    aligned_image2 = aligned_image2[y_min:y_max, x_min:x_max]

    return aligned_image1, aligned_image2, (x_min, y_min), (x_max, y_max)


def compute_intersection(h, w, M):
    """
    Returns the intersection region coordinates of image1 after applying an affine transformation.

    Parameters:
        h (int): Height of image1.
        w (int): Width of image1.
        M (np.ndarray): 2x3 Affine transformation matrix.

    Returns:
        tuple: (x_min, y_min, x_max, y_max) coordinates of the intersection region.
    """
    # Define corners of the image
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
    
    # Apply affine transformation
    transformed = cv2.transform(corners[None], M)[0]

    # Vectorized computation of bounding box
    x_coords = transformed[:, 0]
    y_coords = transformed[:, 1]

    x_min = max(0, int(np.ceil(np.max(x_coords[[0, 2]]))))
    x_max = min(w, int(np.floor(np.min(x_coords[[1, 3]]))))
    y_min = max(0, int(np.ceil(np.max(y_coords[[0, 1]]))))
    y_max = min(h, int(np.floor(np.min(y_coords[[2, 3]]))))

    # Validate intersection region
    if x_max <= x_min or y_max <= y_min:
        print("Error: No valid intersection found.")
        return 0, 0, w, h

    return x_min, y_min, x_max, y_max

def build_motion_boxes(coords, selectionSide):
    """
    Converts motion coordinates into square bounding boxes.

    Parameters:
        coords (List[Tuple[int, int]]): List of (x, y) coordinates of motion points.
        selectionSide (int): Side length of each square bounding box.

    Returns:
        List[Tuple[Tuple[int, int], Tuple[int, int]]]: List of (top_left, bottom_right) box coordinates.
    """
    half_side = selectionSide // 2
    boxes = []

    for x, y in coords:
        top_left = (x - half_side, y - half_side)
        bottom_right = (x + half_side, y + half_side)
        boxes.append((top_left, bottom_right))

    return boxes

def detect_motion(gray1, gray2, keypoints1, descriptors1, keypoints2, descriptors2, s, es, detection_area_left_top, matcher, m):
    """
    Returns coordinates of detected motion points between two frames.

    Parameters:
        gray1, gray2: Grayscale input frames.
        keypoints1, descriptors1: ORB keypoints/descriptors from gray1.
        keypoints2, descriptors2: ORB keypoints/descriptors from gray2.
        s (float): Downscale factor used in alignment.
        m (int): Number of motion points to return.

    Returns:
        List of (x, y) tuples: Coordinates in the original frame space.
    """
    aligned1, aligned2, top_left, _ = align_images(
        gray1, gray2,
        keypoints1, descriptors1,
        keypoints2, descriptors2,
        s, matcher
    )
    diff = cv2.absdiff(aligned1, aligned2)

    coords = []
    if m == 1:
        _, _, _, maxLoc = cv2.minMaxLoc(diff)
        x = detection_area_left_top[0] + int((maxLoc[0] + top_left[0])/es)
        y = detection_area_left_top[1] + int((maxLoc[1] + top_left[1])/es)
        coords.append((x, y))
    else:
        flat = diff.flatten()
        m = min(m, len(flat))
        top_indices = np.argpartition(flat, -m)[-m:]
        top_indices = top_indices[np.argsort(flat[top_indices])[::-1]]

        h, w = diff.shape
        ys, xs = np.unravel_index(top_indices, (h, w))

        for x, y in zip(xs, ys):
            coords.append((detection_area_left_top[0] + int((x + top_left[0])/es), detection_area_left_top[1] + int((y + top_left[1])/es)))

    return coords

def crop_frame(frame, crop_percentage):
    """Crop the central region of the frame based on crop_percentage."""
    height, width = frame.shape[:2]
    crop_factor = crop_percentage / 100.0

    crop_h = int(height * crop_factor)
    crop_w = int(width * crop_factor)

    y1 = (height - crop_h) // 2
    y2 = y1 + crop_h
    x1 = (width - crop_w) // 2
    x2 = x1 + crop_w

    return frame[y1:y2, x1:x2]

def crop_and_resize(frame, crop_percentage, es):
    """Crop frame by percentage and resize with scale `es`."""
    if crop_percentage < 100:
        frame = crop_frame(frame, crop_percentage)
    return cv2.resize(frame, (0, 0), fx=es, fy=es, interpolation=cv2.INTER_AREA)

def play_and_detect(videoFile, start_time, end_time, fpsStep, crop_percentage, es, s, selectionSide=30, m=1,
                    save_output=False, output_file="output_with_motion.avi"):
    # Configure BFMatcher for binary descriptors like ORB
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


    cap = cv2.VideoCapture(videoFile)

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detection_area_left_top = int(frame_width*(1-crop_percentage/100)/2), int(frame_height*(1-crop_percentage/100)/2)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_time = total_frames / fps

    end_time = min(end_time if end_time else total_time, total_time)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))
    current_time = start_time

    # Read and process the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame.")
        return
    
    # Initialize video writer if needed
    writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    # Configure ORB detector
    numOfKeypoints = (frame_width*frame_width) // 10000
    orb = cv2.ORB_create(nfeatures=numOfKeypoints)
    
    prev_frame = crop_and_resize(frame, crop_percentage, es)
    prev_small = cv2.resize(prev_frame, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
    prev_keypoints, prev_descriptors = orb.detectAndCompute(prev_small, None)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    if prev_descriptors is None or len(prev_descriptors) < 2:
        raise ValueError("Not enough keypoints found in the first frame.")

    # Performance monitoring
    N, time_avg = 0, 0.0

    while cap.isOpened() and current_time <= end_time:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(current_time * fps))
        ret, frame = cap.read()
        if not ret:
            break

        # Start detection time
        start_tick = cv2.getTickCount()

        curr_frame = crop_and_resize(frame, crop_percentage, es)
        curr_small = cv2.resize(curr_frame, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
        curr_keypoints, curr_descriptors = orb.detectAndCompute(curr_small, None)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        if curr_descriptors is None or len(curr_descriptors) < 2:
            print("Warning: Not enough keypoints found in the current frame.")
        else:
            objects_coords = detect_motion(
                prev_gray, curr_gray,
                prev_keypoints, prev_descriptors,
                curr_keypoints, curr_descriptors,
                s, es, detection_area_left_top, matcher, m)

        # Update previous values only when detection is successful
        prev_frame, prev_gray = curr_frame, curr_gray
        prev_keypoints, prev_descriptors = curr_keypoints, curr_descriptors

        current_time += fpsStep / fps

        # Time calculation
        end_tick = cv2.getTickCount()
        time_ms = (end_tick - start_tick) / cv2.getTickFrequency() * 1000
        time_avg += (time_ms - time_avg) / (N := N + 1)
        print(f"Average processing time: {time_avg:.2f} ms")

        # Draw detected motion boxes on the current frame
        motion_boxes = build_motion_boxes(objects_coords, selectionSide)
        for tl, br in motion_boxes:
            cv2.rectangle(frame, tl, br, (0, 255, 0), 2)
            
        cv2.imshow("Real-time Object Detection", frame)

        if save_output:
            writer.write(frame)

        if cv2.waitKey(30) & 0xFF == 27:  # ESC key
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

# Run real-time detection
play_and_detect("FullCars.mp4", start_time=38, end_time=88, fpsStep=3, crop_percentage = 75, es=0.5, s=0.5, m=1, save_output=True)