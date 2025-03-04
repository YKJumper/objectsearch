import cv2
import numpy as np
import threading
from queue import Queue

global bitThresh
bitThresh = 40  # Initial threshold value
def on_mouse_scroll(event, x, y, flags, param):
    global bitThresh
    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            bitThresh = min(bitThresh + 5, 255)  # Increase threshold
        else:
            bitThresh = max(bitThresh - 5, 0)  # Decrease threshold
    print(f"bitThresh updated: {bitThresh}")

def get_templates(image, template_size=20):
    """
    Extracts four small templates from the image at specified diagonal positions.
    """
    height, width = image.shape[:2]
    Xc, Yc = width // 2, height // 2
    dX, dY = 3 * width // 8, 3 * height // 8
    
    positions = [
        (Xc - dX, Yc - dY),
        (Xc - dX, Yc + dY),
        (Xc + dX, Yc - dY),
        (Xc + dX, Yc + dY)
    ]
    
    templates = []
    for x, y in positions:
        x1, y1 = max(x - template_size // 2, 0), max(y - template_size // 2, 0)
        x2, y2 = min(x + template_size // 2, width), min(y + template_size // 2, height)
        templates.append((image[y1:y2, x1:x2], (x, y)))
    
    return templates

def process_frames(videoFile, startTime, timeStep, timeDelta, frame_queue, endTime=None, sizeThresh=1):
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
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(T1 * fps))
        ret1, frame1 = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((T1 + timeDelta) * fps))
        ret2, frame2 = cap.read()
        
        if not ret1 or not ret2:
            print("Error: Could not retrieve frames from the video.")
            break
        
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, bitThresh, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > sizeThresh:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        frame_queue.put(frame1)
        T1 += timeStep
    
    cap.release()
    frame_queue.put(None)

def detect_changes(videoFile, startTime, timeStep, timeDelta, endTime=None, displayTime=0.5, sizeThresh=1):
    frame_queue = Queue(maxsize=10)
    processing_thread = threading.Thread(target=process_frames, args=(videoFile, startTime, timeStep, timeDelta, frame_queue, endTime, sizeThresh))
    processing_thread.start()
    
    cv2.namedWindow("Detected Differences")
    cv2.setMouseCallback("Detected Differences", on_mouse_scroll)
    
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        cv2.imshow("Detected Differences", frame)
        if cv2.waitKey(int(displayTime * 1000)) & 0xFF == 27:  # ESC key to exit
            break
    
    cv2.destroyAllWindows()
    processing_thread.join()

detect_changes("pidori.mp4", startTime=0, timeStep=2.0, timeDelta=0.3, endTime=999, displayTime=2.0)
