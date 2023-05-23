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

        vector<int> gazeData = detectAndDisplay(frame);

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

vector<int> detectAndDisplay(Mat frame)
{
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    //-- Detect faces
    vector<Rect> faces;
    face_cascade.detectMultiScale(frame_gray, faces);

    if (faces.empty()) {
        return {-1, -1, -1}; // No face detected
    }

    for (const auto& face : faces)
    {
        //-- In each face, detect eyes
        Mat faceROI = frame(face);
        cvtColor(faceROI, frame_gray, COLOR_BGR2GRAY); // Convert the face ROI to grayscale

        vector<Rect> eyes;
        eyes_cascade.detectMultiScale(frame_gray, eyes);

        if (eyes.empty()) {
            continue; // No eyes detected, move to the next face
        }

        for (const auto& eye : eyes)
        {
            Mat eyeROI = faceROI(eye);
            Point pupil = findPupil(eyeROI);
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
    Mat gray;
    cvtColor(eye, gray, COLOR_BGR2GRAY);

    // Blur the image to reduce noise
    medianBlur(gray, gray, 5);

    // Compute the Scharr gradient
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    // Gradient X
    Scharr(gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);

    // Gradient Y
    Scharr(gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y);

    // Total Gradient (approximate)
    Mat grad;
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    // Threshold and binary the image, black means the pupil region
    Mat bw;
    adaptiveThreshold(~grad, bw, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);

    // Find the center of gravity
    Moments mu = moments(bw, true);
    Point center(mu.m10 / mu.m00, mu.m01 / mu.m00);

    return center;
}
