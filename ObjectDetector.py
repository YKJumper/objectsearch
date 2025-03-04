import cv2
import numpy as np

# Define video file and timestamps (in seconds)
video_path = "pidori.mp4"
T1 = 178  # First frame time in seconds
T2 = 178.3  # Second frame time in seconds

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
    
    cv2.waitKey(int(delay * 10000))
    cv2.destroyAllWindows()

def get_frame_at_time(videoFile, time_sec, crop_percentage=75):
    cap = cv2.VideoCapture(videoFile)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find lightest points (bright objects)
    _, light_thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    light_contours, _ = cv2.findContours(light_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find darkest points (dark objects)
    _, dark_thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    dark_contours, _ = cv2.findContours(dark_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def get_centroids(contours, max_points):
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

    return light_points + dark_points  # Combine light and dark points

def align_images(image1, image2, num_keypoints=5):
    """
    Align image2 to image1 using keypoints and return only the central part of the aligned image.

    :param image1: Reference image (NumPy array)
    :param image2: Image to be aligned (NumPy array)
    :param num_keypoints: Number of keypoints to use for alignment (default: 5)
    :param central_square_percentage: Percentage of the total image area to retain in the center (default: 50%)
    :return: Cropped aligned image, cropped reference image, and transformation matrix
    """
    keypoints1 = find_keypoints(image1, num_keypoints)
    keypoints2 = find_keypoints(image2, num_keypoints)

    if len(keypoints1) < 3 or len(keypoints2) < 3:
        print("Not enough keypoints found for alignment.")
        return image2, image1, np.eye(2, 3, dtype=np.float32)  # Identity transformation if keypoints are insufficient

    # Convert to numpy arrays
    keypoints1 = np.array(keypoints1[:3], dtype=np.float32)  # Need at least 3 points for affine transform
    keypoints2 = np.array(keypoints2[:3], dtype=np.float32)

    # Compute affine transformation matrix
    M = cv2.getAffineTransform(keypoints2, keypoints1)

    # Warp image2 to align with image1
    aligned_image = cv2.warpAffine(image2, M, (image1.shape[1], image1.shape[0]))

    return aligned_image

def find_common_area(image1, M):
    """Find the common region between the aligned images."""
    h, w = image1.shape[:2]

    # Define the corners of image2 in original space
    corners = np.float32([[0, 0], [w, 0], [0, h], [w, h]]).reshape(-1, 1, 2)

    # Transform corners using the transformation matrix
    transformed_corners = cv2.transform(corners, np.vstack([M, [0, 0, 1]]))

    # Find the intersection of bounding boxes
    x_min = max(0, int(min(transformed_corners[:, 0, 0])))
    y_min = max(0, int(min(transformed_corners[:, 0, 1])))
    x_max = min(w, int(max(transformed_corners[:, 0, 0])))
    y_max = min(h, int(max(transformed_corners[:, 0, 1])))

    return x_min, y_min, x_max, y_max

# Define a function to check if two bounding boxes are close
def are_boxes_close(box1, box2, threshold=10):
    """
    Check if two bounding boxes are close based on a pixel distance threshold.
    
    :param box1: Tuple (x, y, w, h) for the first bounding box
    :param box2: Tuple (x, y, w, h) for the second bounding box
    :param threshold: Maximum distance between box centers to consider them close
    :return: True if close, False otherwise
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Compute center points
    center1 = (x1 + w1 // 2, y1 + h1 // 2)
    center2 = (x2 + w2 // 2, y2 + h2 // 2)
    
    # Compute Euclidean distance
    distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
    
    return distance <= threshold

# Capture frames at T1 and T2
image1 = get_frame_at_time(video_path, T1, crop_percentage=71)
image2 = get_frame_at_time(video_path, T2, crop_percentage=71)

alligned_image = align_images(image1, image2)

image1 = alligned_image[0]
image2 = alligned_image[1]
# Ensure frames were successfully extracted
if image1 is None or image2 is None:
    print("Error: Could not retrieve frames from the video.")
    exit()
    
# Align image2 to image1
# aligned_image1, aligned_image2, M = align_images(image1, image2)

# Find the common part
# x_min, y_min, x_max, y_max = find_common_area(aligned_image1, aligned_image2, M)

# Crop both images to the common region
# cropped_image1 = aligned_image1[y_min:y_max, x_min:x_max]
# cropped_image2 = aligned_image2[y_min:y_max, x_min:x_max]

# Display results
# resize_and_display(cropped_image2, title="Aligned Image")
# resize_and_display(cropped_image1, title="Reference Image Cropped")
# resize_and_display(cv2.absdiff(cropped_image1, cropped_image2), title="Detected Difference")

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Compute absolute difference
diff = cv2.absdiff(gray1, gray2)
resize_and_display(cv2.absdiff(gray1, gray2), title="Grey-coloured detected Difference")

# Find lightest points (bright objects)
_, light_thresh = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY)
# Find darkest points (dark objects)
_, dark_thresh = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY_INV)

# Find contours of changed regions
light_contours, _ = cv2.findContours(light_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
dark_contours, _ = cv2.findContours(dark_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

boxSizeThreshold = 10
# Extract bounding boxes for light and dark contours
light_boxes = [cv2.boundingRect(contour) for contour in light_contours if cv2.contourArea(contour) > boxSizeThreshold]
dark_boxes = [cv2.boundingRect(contour) for contour in dark_contours if cv2.contourArea(contour) > boxSizeThreshold]
# Draw bounding boxes for overlapping or close differences

for box in light_boxes:
    x, y, w, h = box
    cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw in green
# boxSizeThreshold = 10
# # Extract bounding boxes for light and dark contours
# light_boxes = [cv2.boundingRect(contour) for contour in light_contours if cv2.contourArea(contour) > boxSizeThreshold]
# dark_boxes = [cv2.boundingRect(contour) for contour in dark_contours if cv2.contourArea(contour) > boxSizeThreshold]
# # Draw bounding boxes for overlapping or close differences
# for light_box in light_boxes:
#     for dark_box in dark_boxes:
#         if are_boxes_close(light_box, dark_box, threshold=15):  # Adjust threshold as needed
#             x, y, w, h = light_box
#             cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw in green

# Display the result


resize_and_display(image1, title="Detected Difference with Bounding Boxes")
