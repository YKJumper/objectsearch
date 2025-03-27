import cv2
import numpy as np
import time
from collections import deque

# Global parameters
bitBrightSelector = 0.65
bitThresh = 40
kpGraphRigidity = 2
numOfKeypoints = 500

# GPU check
use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0

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

def highlight_motion(frame1, frame2, bbox_history=None, max_history=15, sizeThresh=1):
    global bitBrightSelector
    global bitThresh
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    if use_cuda:
        gray1_gpu = cv2.cuda_GpuMat()
        gray2_gpu = cv2.cuda_GpuMat()
        gray1_gpu.upload(gray1)
        gray2_gpu.upload(gray2)
        diff_gpu = cv2.cuda.absdiff(gray1_gpu, gray2_gpu)
        diff = diff_gpu.download()
    else:
        aligned1, aligned2, top_left, right_bottom = align_images(gray1, gray2)
        diff = cv2.absdiff(aligned1, aligned2)

    minVal, maxVal, _, maxLoc = cv2.minMaxLoc(diff)
    # bitThresh = int(bitBrightSelector * (maxVal - minVal) + minVal)
    # _, thresh = cv2.threshold(diff, bitThresh, 255, cv2.THRESH_BINARY)
    # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # Keep a history of bounding boxes to smooth motion
    # boxes = []
    # for contour in contours:
    #     if cv2.contourArea(contour) > sizeThresh:
    #         x, y, w, h = cv2.boundingRect(contour)
    #         x, y = x + top_left[0], y + top_left[1]
    #         boxes.append((x, y, w, h))

    # if bbox_history is not None:
    #     bbox_history.append(boxes)
    #     if len(bbox_history) > max_history:
    #         bbox_history.popleft()

    #     # Compute average boxes
    #     smoothed_boxes = []
    #     for i in range(len(bbox_history[0])):
    #         coords = np.array([b[i] for b in bbox_history if len(b) > i])
    #         avg = np.mean(coords, axis=0).astype(int)
    #         smoothed_boxes.append(tuple(avg))
    #     boxes = smoothed_boxes

    annotated_frame = frame2.copy()
    # for (x, y, w, h) in boxes:
    #     cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Mark strongest change
    selectionSide = 30
    cv2.rectangle(annotated_frame,
                  (maxLoc[0] - selectionSide // 2, maxLoc[1] - selectionSide // 2),
                  (maxLoc[0] + selectionSide // 2, maxLoc[1] + selectionSide // 2),
                  (0, 255, 0), 2)
    return annotated_frame

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

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Cannot read initial frame.")
        return

    bbox_history = deque()

    while cap.isOpened() and current_time <= end_time:
        frame_number = int(current_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, curr_frame = cap.read()
        if not ret:
            break

        annotated_frame = highlight_motion(prev_frame, curr_frame, bbox_history)

        cv2.imshow("Real-time Object Detection", annotated_frame)
        if save_output:
            writer.write(annotated_frame)

        key = cv2.waitKey(int(1000 / fps))
        if key == 27:  # ESC
            break

        prev_frame = curr_frame
        current_time += 1 / fps

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

# Play from 33s to 60s, enable GPU, smoothing, and save output
play_and_detect("Stadium.mp4", start_time=0, end_time=80, save_output=True)

# "orlan.mp4", start_time=11,
# "FullCars.mp4", start_time=35,
# "pidor2.mp4", start_time=0,
# "stableBalcony.mp4", start_time=18,
# "wavedBalcony.mp4", start_time=0,
# "Stadium.mp4", start_time=0,
