#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

tuple<int, int, int, int> compute_intersection(int h, int w, const Mat& M) {
    // Define corners of the image
    vector<Point2f> corners = {Point2f(0, 0), Point2f(w, 0), Point2f(0, h), Point2f(w, h)};
    
    // Apply affine transformation
    std::vector<cv::Point2f> transformed;
    transform(corners, transformed, M);
    cout << "asdasd " << transformed[0].x << " " << transformed[1].x << " " << transformed[2].x << " " << transformed[3].x << endl;
    cout << "asdasd " << transformed[0].y << " " << transformed[1].y << " " << transformed[2].y << " " << transformed[3].y << endl;

    // Vectorized computation of bounding box
    float x_min = max(0.f, min(transformed[0].x, transformed[2].x));
    float x_max = min((float)w, max(transformed[1].x, transformed[3].x));
    float y_min = max(0.f, min(transformed[0].y, transformed[1].y));
    float y_max = min((float)h, max(transformed[2].y, transformed[3].y));

    // Validate intersection region
    if (x_max <= x_min || y_max <= y_min) {
        cout << "Error: No valid intersection found." << endl;
        return {0, 0, w, h};
    }
    return {static_cast<int>(x_min), static_cast<int>(y_min), static_cast<int>(x_max), static_cast<int>(y_max)};
}

void printMatType(const Mat& mat) {
    int type = mat.type();
    
    // Get the depth
    string depth;
    switch (type & CV_MAT_DEPTH_MASK) {
        case CV_8U:  depth = "8U"; break;
        case CV_8S:  depth = "8S"; break;
        case CV_16U: depth = "16U"; break;
        case CV_16S: depth = "16S"; break;
        case CV_32S: depth = "32S"; break;
        case CV_32F: depth = "32F"; break;
        case CV_64F: depth = "64F"; break;
        default: depth = "Unknown"; break;
    }
    
    // Get the number of channels
    int channels = (type >> CV_CN_SHIFT) & 0x3F;
    
    cout << "Matrix type: " << depth << " | Channels: " << channels + 1 << endl;
}

void printGoodMatches(const std::vector<cv::DMatch>& goodMatches) {
    std::cout << "Total good matches: " << goodMatches.size() << std::endl;
    for (size_t i = 0; i < goodMatches.size(); i++) {
        std::cout << "Match " << i << ": Distance=" << goodMatches[i].distance
                  << ", QueryIdx=" << goodMatches[i].queryIdx
                  << ", TrainIdx=" << goodMatches[i].trainIdx << std::endl;
    }
}

void printPoints(const std::vector<cv::Point2f>& points, const std::string& name) {
    std::cout << name << ":\n";
    for (size_t i = 0; i < points.size(); i++) {
        std::cout << "Point " << i << ": (" << points[i].x << ", " << points[i].y << ")\n";
    }
}

void alignImages(const Mat& image1, const Mat& image2,
                 vector<KeyPoint>& keypoints1, Mat& descriptors1,
                 vector<KeyPoint>& keypoints2, Mat& descriptors2,
                 double s, Mat& alignedImage1, Mat& alignedImage2,
                 Point& leftTop, Point& rightBottom) {
    int h = image1.rows;
    int w = image1.cols;

    // Configure FLANN matcher
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // Apply Lowe's ratio test
    vector<DMatch> goodMatches;
    for (const auto& match : matches) {
        // You can adjust the threshold here based on the min and max distances
        if (match.distance <= 60) {
            goodMatches.push_back(match);
        }
    }
    //printGoodMatches(goodMatches);
    if (goodMatches.size() < 10) return;

    vector<Point2f> srcPoints, dstPoints;
    for (const auto& match : goodMatches) {
        srcPoints.push_back(keypoints2[match.trainIdx].pt);
        dstPoints.push_back(keypoints1[match.queryIdx].pt);
    }

    Mat M = estimateAffine2D(srcPoints, dstPoints, noArray(), RANSAC, 5.0);
    //printPoints(srcPoints, "Source Points");
    //printPoints(dstPoints, "Destination Points");
    if (M.empty()) return;

    M.at<double>(0, 2) /= s;
    M.at<double>(1, 2) /= s;

    warpAffine(image2, alignedImage2, M, Size(w, h));

    auto [x_min, y_min, x_max, y_max] = compute_intersection(h, w, M);
    cout << x_min << " " << y_min << " " << x_max <<  " " << y_max << " " << endl;

    // Crop both images to the overlapping region
    Rect roi(x_min, y_min, x_max - x_min, y_max - y_min);

    alignedImage1 = image1(roi);

    alignedImage2 = alignedImage2(roi);


    // Set the left top and right bottom coordinates
    leftTop = Point(x_min, y_min);
    rightBottom = Point(x_max, y_max);
}

vector<Point> detectMotion(const Mat& gray1, const Mat& gray2,
                           vector<KeyPoint>& keypoints1, Mat& descriptors1,
                           vector<KeyPoint>& keypoints2, Mat& descriptors2,
                           double s, double es, Point detectionAreaTopLeft) {
    Mat aligned1, aligned2;
    Point topLeft, bottomRight;
    alignImages(gray1, gray2, keypoints1, descriptors1, keypoints2, descriptors2, s,
                aligned1, aligned2, topLeft, bottomRight);

    Mat diff;
    absdiff(aligned1, aligned2, diff);
    cv::imwrite("/home/dmytro/object_movement/res1.png", diff);
    cout << topLeft.x << " aaa " << topLeft.y << endl;

    vector<Point> coords;
    Point maxLoc;
    minMaxLoc(diff, nullptr, nullptr, nullptr, &maxLoc);
    Point res(detectionAreaTopLeft.x + (maxLoc.x + topLeft.x) / es,
                detectionAreaTopLeft.y + (maxLoc.y + topLeft.y) / es);
    coords.push_back(res);

    cout << res.x << " abcc " << res.y << endl;

    return coords;
}

Mat cropFrame(const Mat& frame, double cropPercentage) {
    if (frame.empty()) {
        cerr << "Error: Empty frame provided." << endl;
        return Mat();
    }

    // Get image dimensions
    int height = frame.rows;
    int width = frame.cols;

    // Compute crop dimensions
    double cropFactor = cropPercentage / 100.0;
    int cropH = static_cast<int>(height * cropFactor);
    int cropW = static_cast<int>(width * cropFactor);

    // Compute crop boundaries
    int y1 = (height - cropH) / 2;
    int y2 = y1 + cropH;
    int x1 = (width - cropW) / 2;
    int x2 = x1 + cropW;

    // Crop and return the central region
    return frame(Range(y1, y2), Range(x1, x2)).clone();
}

Mat cropAndResize(Mat frame, double cropPercentage, double scaleFactor) {
    if (cropPercentage < 100) {
        frame = cropFrame(frame, cropPercentage);
    }
    // Resize the image using the scaling factor
    Mat resizedFrame;
    resize(frame, resizedFrame, Size(), scaleFactor, scaleFactor, INTER_AREA);
    return resizedFrame;
}

int playAndDetect(const string& videoFile, int start_time, int end_time, int fpsStep,
                   int crop_percentage, double es, double s, int numOfKeypoints) {
    Ptr<ORB> orb = ORB::create(numOfKeypoints);
                
    // Open video
    VideoCapture cap(videoFile);
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open video." << endl;
        return -1;
    }
                
    // Get video properties
    double fps = cap.get(CAP_PROP_FPS);
    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    float percentage = 1.0 - float(crop_percentage / 100.0);
    Point detection_area_left_top(int(frame_width * percentage) / 2, int(frame_height * percentage) / 2);
                
    int total_frames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));
    double total_time = total_frames / fps;
    cout << frame_width << " " << frame_height << " " << detection_area_left_top.x << " " << detection_area_left_top.y << " " << total_frames << " " << total_time << endl;
                
    // Calculate end time
    cap.set(CAP_PROP_POS_FRAMES, static_cast<int>(start_time * fps));
                
    double current_time = start_time;
                
    // Read the first frame
    Mat frame;
    bool ret = cap.read(frame);
    if (!ret) {
        cerr << "Error: Cannot read first frame." << endl;
        return -1;
    }
                
    // Process the first frame
    Mat prev_frame = cropAndResize(frame, crop_percentage, es);
    cout << prev_frame.cols << " pref " << prev_frame.rows << endl;
    Mat prev_small;
    resize(prev_frame, prev_small, Size(), s, s, INTER_AREA);
                
    // Detect keypoints and descriptors
    vector<KeyPoint> prev_keypoints;
    Mat prev_descriptors;
    orb->detectAndCompute(prev_small, noArray(), prev_keypoints, prev_descriptors);
    cvtColor(prev_frame, prev_frame, COLOR_BGR2GRAY);
    if (prev_descriptors.empty() || prev_descriptors.rows < 2) {
        cerr << "Not enough keypoints found in the first frame." << endl;
        return -1;
    }
                
    // Performance monitoring
    int N = 0;
    int frameCounter = 0;
    double time_avg = 0.0;
    while (cap.isOpened() && current_time <= end_time) {
        cap.set(CAP_PROP_POS_FRAMES, static_cast<int>(current_time * fps));
        Mat frame, gray;
        cap >> frame;
        if (frame.empty()) break;

        double startTick = getTickCount();
        Mat cur_frame = cropAndResize(frame, crop_percentage, es);
        Mat cur_small;
        resize(cur_frame, cur_small, Size(), s, s, INTER_AREA);
        cvtColor(cur_frame, gray, COLOR_BGR2GRAY);

        vector<KeyPoint> currKeypoints;
        Mat currDescriptors;
        orb->detectAndCompute(cur_small, noArray(), currKeypoints, currDescriptors);

        if (!currDescriptors.empty() && currDescriptors.rows >= 2) {
           vector<Point> motionCoords = detectMotion(prev_frame, gray, prev_keypoints, prev_descriptors,
                                                      currKeypoints, currDescriptors, s, es,
                                                      detection_area_left_top);
            for (const auto& pt : motionCoords) {
                rectangle(frame, Rect(pt.x - 15, pt.y - 15, 30, 30), Scalar(0, 255, 0), 2);
            }
        }
        if (frameCounter == 2000)
        {
            break;
        }

        prev_frame = gray.clone();
        prev_keypoints = currKeypoints;
        prev_descriptors = currDescriptors.clone();

        imshow("Real-time Object Detection", frame);
        if (waitKey(30) == 27) break;

        double endTick = getTickCount();
        time_avg += ((endTick - startTick) / getTickFrequency() * 1000 - time_avg) / (++frameCounter);
        cout << "Average processing time: " << time_avg << " ms" << endl;
        current_time += fpsStep / fps;
    }
    cap.release();
    destroyAllWindows();
    return 1;
}

int main() {
    playAndDetect("/home/dmytro/object_movement/FullCars.mp4", 39, 388, 3, 75, 0.5, 0.5, 250);
    return 0;
}
