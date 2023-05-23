#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

/** Function Headers */
vector<int> detectAndDisplay(Mat frame);
Point findPupil(Mat eye);

/** Global variables */
String face_cascade_name, eyes_cascade_name;
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

int main(int argc, const char** argv)
{
    face_cascade_name = "data/haarcascade_frontalface_alt.xml";
    eyes_cascade_name = "data/haarcascade_eye_tree_eyeglasses.xml";

    if (!face_cascade.load(face_cascade_name) || !eyes_cascade.load(eyes_cascade_name))
    {
        cout << "--(!)Error loading cascades\n";
        return -1;
    }

    int camera_device = 0;
    VideoCapture capture;
    capture.open(camera_device);
    cv::namedWindow("Eyes Tracker", cv::WINDOW_NORMAL);

    if (!capture.isOpened())
    {
        cout << "--(!)Error opening video capture\n";
        return -1;
    }

    Mat frame;
    while (capture.read(frame))
    {
        if (frame.empty())
        {
            cout << "--(!) No captured frame -- Break!\n";
            break;
        }

        // Convert frame to grayscale
        Mat frame_gray;
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

        vector<int> gazeData = detectAndDisplay(frame_gray);

        imshow("Eyes Tracker", frame);
        if (waitKey(1) == 27)
        {
            break; // escape
        }

        cout << "OUTPUT:\n  [";
        for (auto i = gazeData.begin(); i != gazeData.end(); i++)
        {
            cout << *i;
            if (i + 1 != gazeData.end())
                cout << ", ";
        }
        cout << "]\n";
    }

    return 0;
}


vector<int> detectAndDisplay(Mat frame_gray)
{
    //-- Convert frame_gray to HUV color space
    Mat frame_hsv;
    cvtColor(frame_gray, frame_hsv, COLOR_GRAY2BGR);
    cvtColor(frame_hsv, frame_hsv, COLOR_BGR2HSV);

    //-- Define lower and upper thresholds for dark eye color in HUV space
    Scalar lower_threshold(0, 0, 0); // Lower threshold for dark eye color
    Scalar upper_threshold(180, 255, 30); // Upper threshold for dark eye color

    //-- Apply the HUV filter to isolate dark eye color regions
    Mat filtered_frame;
    inRange(frame_hsv, lower_threshold, upper_threshold, filtered_frame);

    //-- Detect faces
    vector<Rect> faces;
    face_cascade.detectMultiScale(frame_gray, faces);

    if (faces.empty()) {
        return {-1, -1, -1}; // No face detected
    }

    for (const auto& face : faces)
    {
        //-- In each face, detect eyes
        Mat faceROI = frame_gray(face);
        Mat filtered_faceROI = filtered_frame(face);

        vector<Rect> eyes;
        eyes_cascade.detectMultiScale(faceROI, eyes);

        if (eyes.empty()) {
            continue; // No eyes detected, move to the next face
        }

        for (const auto& eye : eyes)
        {
            Mat eyeROI = faceROI(eye);
            Mat filtered_eyeROI = filtered_faceROI(eye);

            Point pupil = findPupil(filtered_eyeROI);
            Point gazeDirection = Point(eyeROI.cols / 2, eyeROI.rows / 2) - pupil;

            int gaze = 0;
            if (abs(gazeDirection.x) > abs(gazeDirection.y))
            {
                // Horizontal direction
                gaze = gazeDirection.x > 0 ? 3 : 4; // 3: Left, 4: Right
            }
            else
            {
                // Vertical direction
                gaze = gazeDirection.y > 0 ? 1 : 2; // 1: Up, 2: Down
            }

            return {pupil.x, pupil.y, gaze};
        }
    }

    return {-1, -1, -1}; // Return -1 when no eye detected
}


Point findPupil(Mat eye)
{
    // Apply adaptive thresholding
    Mat bw;
    adaptiveThreshold(eye, bw, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 15, 10);

    // Apply morphological operations for noise removal
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(bw, bw, MORPH_OPEN, kernel);

    // Find the center of gravity
    Moments mu = moments(bw, true);
    Point center(mu.m10 / mu.m00, mu.m01 / mu.m00);

    return center;
}
