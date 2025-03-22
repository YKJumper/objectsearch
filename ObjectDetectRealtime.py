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
    return image1, aligned_image2, (0, 0)

def highlight_motion(frame1, frame2):
    global bitThresh
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    aligned1, aligned2, offset = align_images(gray1, gray2)
    diff = cv2.absdiff(aligned1, aligned2)

    minVal, maxVal, _, maxLoc = cv2.minMaxLoc(diff)
    bitThresh = int(bitBrightSelector * (maxVal - minVal) + minVal)
    _, thresh = cv2.threshold(diff, bitThresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated_frame = frame2.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 1:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Highlight max diff location
    selectionSide = 30
    cv2.rectangle(annotated_frame,
                  (maxLoc[0] - selectionSide // 2, maxLoc[1] - selectionSide // 2),
                  (maxLoc[0] + selectionSide // 2, maxLoc[1] + selectionSide // 2),
                  (0, 255, 0), 2)
    return annotated_frame

def play_and_detect(videoFile):
    cap = cv2.VideoCapture(videoFile)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame.")
        return

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        detected_frame = highlight_motion(prev_frame, curr_frame)

        cv2.imshow("Real-time Object Detection", detected_frame)
        if cv2.waitKey(30) & 0xFF == 27:  # ESC key to stop
            break

        prev_frame = curr_frame

    cap.release()
    cv2.destroyAllWindows()

# Run real-time detection
play_and_detect("FullCars.mp4")
