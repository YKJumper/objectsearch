#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <numeric>

using namespace cv;
using namespace std;

Rect compute_intersection(int h, int w, const Mat& M) {
    vector<Point2f> corners = { {0, 0}, {(float)w, 0}, {0, (float)h}, {(float)w, (float)h} };
    vector<Point2f> transformed;
    transform(corners, transformed, M);

    float x_min = max(0.f, ceil(max(transformed[0].x, transformed[2].x)));
    float x_max = min((float)w, floor(min(transformed[1].x, transformed[3].x)));
    float y_min = max(0.f, ceil(max(transformed[0].y, transformed[1].y)));
    float y_max = min((float)h, floor(min(transformed[2].y, transformed[3].y)));

    if (x_max <= x_min || y_max <= y_min) {
        cerr << "Error: No valid intersection found." << endl;
        return Rect(0, 0, w, h);
    }

    return Rect((int)x_min, (int)y_min, (int)(x_max - x_min), (int)(y_max - y_min));
}

tuple<Mat, Mat, Point, Point> align_images(
    const Mat& image1, const Mat& image2,
    const vector<KeyPoint>& keypoints1, const Mat& descriptors1,
    const vector<KeyPoint>& keypoints2, const Mat& descriptors2,
    float s
) {
    int h = image1.rows;
    int w = image1.cols;

    // Use FLANN with LSH
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    Mat desc1f, desc2f;
    descriptors1.convertTo(desc1f, CV_32F);
    descriptors2.convertTo(desc2f, CV_32F);

    vector<vector<DMatch>> matches;
    matcher->knnMatch(desc1f, desc2f, matches, 2);

    vector<DMatch> good_matches;
    for (auto& pair : matches) {
        if (pair.size() == 2 && pair[0].distance < 0.75f * pair[1].distance) {
            good_matches.push_back(pair[0]);
        }
    }

    if (good_matches.size() < 10) {
        return { image1.clone(), image2.clone(), Point(0, 0), Point(0, 0) };
    }

    vector<Point2f> src_pts, dst_pts;
    for (const auto& match : good_matches) {
        src_pts.push_back(keypoints2[match.trainIdx].pt);
        dst_pts.push_back(keypoints1[match.queryIdx].pt);
    }

    Mat M_small = estimateAffine2D(src_pts, dst_pts, noArray(), RANSAC, 5.0);
    if (M_small.empty()) {
        return { image1.clone(), image2.clone(), Point(0, 0), Point(0, 0) };
    }

    M_small.at<double>(0, 2) /= s;
    M_small.at<double>(1, 2) /= s;

    Mat aligned_image2;
    warpAffine(image2, aligned_image2, M_small, Size(w, h));

    Rect roi = compute_intersection(h, w, M_small);
    Mat aligned_image1 = image1(roi).clone();
    aligned_image2 = aligned_image2(roi).clone();

    return { aligned_image1, aligned_image2, roi.tl(), roi.br() };
}

vector<pair<Point, Point>> build_motion_boxes(const vector<Point>& coords, int selectionSide) {
    int half_side = selectionSide / 2;
    vector<pair<Point, Point>> boxes;
    for (const auto& p : coords) {
        boxes.emplace_back(Point(p.x - half_side, p.y - half_side), Point(p.x + half_side, p.y + half_side));
    }
    return boxes;
}

vector<Point> detect_motion(
    const Mat& gray1, const Mat& gray2,
    const vector<KeyPoint>& keypoints1, const Mat& descriptors1,
    const vector<KeyPoint>& keypoints2, const Mat& descriptors2,
    float s, float es, Point detection_area_left_top, int m = 1
) {
    auto [aligned1, aligned2, top_left, _] = align_images(gray1, gray2, keypoints1, descriptors1, keypoints2, descriptors2, s);

    Mat diff;
    absdiff(aligned1, aligned2, diff);

    vector<Point> coords;
    if (m == 1) {
        Point maxLoc;
        minMaxLoc(diff, nullptr, nullptr, nullptr, &maxLoc);
        int x = detection_area_left_top.x + static_cast<int>((maxLoc.x + top_left.x) / es);
        int y = detection_area_left_top.y + static_cast<int>((maxLoc.y + top_left.y) / es);
        coords.emplace_back(x, y);
    } else {
        Mat flat = diff.reshape(1, 1);
        vector<uchar> vec(flat.begin<uchar>(), flat.end<uchar>());
        vector<int> indices(vec.size());
        iota(indices.begin(), indices.end(), 0);

        partial_sort(indices.begin(), indices.begin() + m, indices.end(), [&](int a, int b) {
            return vec[a] > vec[b];
        });

        int h = diff.rows, w = diff.cols;
        for (int i = 0; i < m; ++i) {
            int idx = indices[i];
            int y = idx / w, x = idx % w;
            coords.emplace_back(
                detection_area_left_top.x + static_cast<int>((x + top_left.x) / es),
                detection_area_left_top.y + static_cast<int>((y + top_left.y) / es)
            );
        }
    }
    return coords;
}

Mat crop_frame(const Mat& frame, float crop_percentage) {
    int height = frame.rows, width = frame.cols;
    float crop_factor = crop_percentage / 100.0;
    int crop_h = static_cast<int>(height * crop_factor);
    int crop_w = static_cast<int>(width * crop_factor);
    int y1 = (height - crop_h) / 2, x1 = (width - crop_w) / 2;
    return frame(Rect(x1, y1, crop_w, crop_h)).clone();
}

Mat crop_and_resize(const Mat& frame, float crop_percentage, float es) {
    Mat cropped = crop_percentage < 100.0f ? crop_frame(frame, crop_percentage) : frame.clone();
    Mat resized;
    resize(cropped, resized, Size(), es, es, INTER_AREA);
    return resized;
}

int main() {
    string videoFile = "FullCars.mp4";
    float start_time = 38.0f, end_time = 388.0f, fpsStep = 4.0f;
    float crop_percentage = 60.0f, es = 0.5f, s = 0.5f;
    int numOfKeypoints = 250, selectionSide = 30;

    Ptr<ORB> orb = ORB::create(numOfKeypoints);
    VideoCapture cap(videoFile);

    if (!cap.isOpened()) {
        cerr << "Error: Cannot open video." << endl;
        return -1;
    }

    double fps = cap.get(CAP_PROP_FPS);
    int frame_width = int(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = int(cap.get(CAP_PROP_FRAME_HEIGHT));
    Point detection_area_left_top(int(frame_width * (1 - crop_percentage / 100) / 2), int(frame_height * (1 - crop_percentage / 100) / 2));
    int total_frames = int(cap.get(CAP_PROP_FRAME_COUNT));
    float total_time = total_frames / fps;
    end_time = min(end_time, total_time);
    cap.set(CAP_PROP_POS_FRAMES, int(start_time * fps));
    float current_time = start_time;

    Mat frame;
    cap >> frame;
    if (frame.empty()) {
        cerr << "Error: Cannot read first frame." << endl;
        return -1;
    }

    Mat prev_frame = crop_and_resize(frame, crop_percentage, es);
    Mat prev_small;
    resize(prev_frame, prev_small, Size(), s, s, INTER_AREA);
    vector<KeyPoint> prev_keypoints;
    Mat prev_descriptors;
    orb->detectAndCompute(prev_small, noArray(), prev_keypoints, prev_descriptors);
    Mat prev_gray;
    cvtColor(prev_frame, prev_gray, COLOR_BGR2GRAY);

    if (prev_descriptors.empty()) {
        cerr << "Error: Not enough keypoints in the first frame." << endl;
        return -1;
    }

    int N = 0;
    double time_avg = 0.0;

    while (cap.isOpened() && current_time <= end_time) {
        cap.set(CAP_PROP_POS_FRAMES, int(current_time * fps));
        cap >> frame;
        if (frame.empty()) break;

        int64 start_tick = getTickCount();

        Mat curr_frame = crop_and_resize(frame, crop_percentage, es);
        Mat curr_small;
        resize(curr_frame, curr_small, Size(), s, s, INTER_AREA);
        vector<KeyPoint> curr_keypoints;
        Mat curr_descriptors;
        orb->detectAndCompute(curr_small, noArray(), curr_keypoints, curr_descriptors);
        Mat curr_gray;
        cvtColor(curr_frame, curr_gray, COLOR_BGR2GRAY);

        vector<Point> objects_coords;
        if (!curr_descriptors.empty()) {
            objects_coords = detect_motion(prev_gray, curr_gray, prev_keypoints, prev_descriptors, curr_keypoints, curr_descriptors, s, es, detection_area_left_top);
        }

        prev_frame = curr_frame.clone();
        prev_gray = curr_gray.clone();
        prev_keypoints = curr_keypoints;
        prev_descriptors = curr_descriptors;

        current_time += fpsStep / fps;
        int64 end_tick = getTickCount();
        double time_ms = (end_tick - start_tick) * 1000.0 / getTickFrequency();
        time_avg += (time_ms - time_avg) / (++N);
        cout << "Average processing time: " << time_avg << " ms" << endl;

        vector<pair<Point, Point>> motion_boxes = build_motion_boxes(objects_coords, selectionSide);
        for (auto& box : motion_boxes) {
            rectangle(frame, box.first, box.second, Scalar(0, 255, 0), 2);
        }

        imshow("Real-time Object Detection", frame);
        if (waitKey(30) == 27) break; // ESC
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
