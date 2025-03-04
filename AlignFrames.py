import cv2
import numpy as np
import threading
from queue import Queue


global bitBrightSelector
global bitThresh
bitThresh = 40  # Initial threshold value
bitBrightSelector = 0.75 # Initial bright selector value

def get_templates_grid(image, template_size=20, grid_size=[4,4]):
    """
    Extracts four small templates from the image at the specified grid positions.
    """
    height, width = image.shape[:2]
    hStep, wStep = height//(grid_size[0]+1), width//(grid_size[1]+1)
    templates = []
    for i in range(1, grid_size[0]+1):
        for j in range(1, grid_size[1]+1):
            x, y = j*wStep, i*hStep
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + template_size, width), min(y + template_size, height)
            templates.append((image[y1:y2, x1:x2], (x, y)))

    return templates

def find_template_positions(image, templates):
    """
    Finds the best match locations of given templates in the reference image.
    """
    positions = []
    for template, original_position in templates:
        res = cv2.matchTemplate(image, template[0], cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        positions.append(min_loc)
    return positions

def dist(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

def align_images(image1, image2, template_size=25):
    """
    Aligns image2 to image1 using templates and returns the aligned image.
    """
    templates = get_templates_grid(image2, template_size, grid_size=[2,2])
    templates_ver = get_templates_grid(image2, int(template_size*1.41), grid_size=[2,2])

    template_positions = find_template_positions(image1, templates)
    template_positions_ver = find_template_positions(image1, templates_ver)

    matched_positions = []
    matched_templates = [] 
    for i in range(len(templates)):
        pos_dst = dist(template_positions[i], template_positions_ver[i])
        if pos_dst<=template_size/4:
            matched_positions.append(template_positions[i])
            matched_templates.append(templates[i])

    if len(matched_positions) < 4:
        print(f"{len(matched_positions)} found. Not enough points for transformation.")
        return image2
    else:
        print(f"{len(matched_positions)} points for transformation were found.")
    
    src_pts = np.array([t[1] for t in matched_templates], dtype=np.float32)
    dst_pts = np.array(matched_positions, dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    aligned_image = cv2.warpPerspective(image2, M, (image1.shape[1], image1.shape[0]))
    
    return aligned_image

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
        frame1 = get_frame_at_time(cap, fps, T1)
        frame2 = get_frame_at_time(cap, fps, T1 + timeDelta)

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        aligned_image = align_images(gray1, gray2)
        diff = cv2.absdiff(aligned_image, gray1)
        
        # resize_and_display(diff, delay=2.5)
        
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
                cv2.rectangle(frame2, (x-wext, y-hext), (x + w + wext, y + h + hext), (0, 0, 255), 2)
        
        frame_queue.put(frame2)
        T1 += timeStep
    
    cap.release()
    frame_queue.put(None)  
    
def resize_and_display(image, screen_width=1920, screen_height=1080, title="Detected Difference", delay=0.5):
    global bitThresh
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
    cv2.waitKey(int(delay * 10000))
    cv2.destroyAllWindows()

def display_frames(frame_queue, DisplayTime=0.5):
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

detect_changes("pidori.mp4", startTime=178, timeStep=1.0, timeDelta=0.15, endTime=999, displayTime=1.0, sizeThresh=1)
