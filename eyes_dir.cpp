#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <SDL2/SDL.h>

using namespace std;
using namespace cv;

/** Function Headers */
vector<int> detectAndDisplay(Mat frame);
Point findPupil(Mat eye);
void updateMousePosition(const Point& gazeDirection, int screenWidth, int screenHeight);

/** Global variables */
String face_cascade_name, eyes_cascade_name;
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
int mouseX = 0;
int mouseY = 0;
Point previousPupil(-1, -1); // Coordonnées précédentes de la pupille

int main(int argc, const char** argv)
{
    face_cascade_name = "data/haarcascade_frontalface_alt.xml";
    eyes_cascade_name = "data/haarcascade_eye_tree_eyeglasses.xml";

    if (!face_cascade.load(face_cascade_name) || !eyes_cascade.load(eyes_cascade_name))
    {
        cout << "--(!)Error loading cascades\n";
        return -1;
    }

    // Initialize SDL for mouse events
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        cout << "Failed to initialize SDL: " << SDL_GetError() << endl;
        return -1;
    }

    // Get the screen information
    SDL_DisplayMode screenMode;
    if (SDL_GetDesktopDisplayMode(0, &screenMode) != 0)
    {
        cout << "Failed to get screen information: " << SDL_GetError() << endl;
        SDL_Quit();
        return -1;
    }

    // Calculate the center coordinates
    int centerX = screenMode.w / 2;
    int centerY = screenMode.h / 2;

    int camera_device = 0;
    VideoCapture capture;
    capture.open(camera_device);
    cv::namedWindow("Eyes Tracker", cv::WINDOW_NORMAL);

    if (!capture.isOpened())
    {
        cout << "--(!)Error opening video capture\n";
        SDL_Quit();
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

        // Update mouse position based on gaze direction
        Point gazeDirection(gazeData[0], gazeData[1]);
        updateMousePosition(gazeDirection, screenMode.w, screenMode.h);
    }

    // Cleanup and quit SDL
    SDL_Quit();

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

    Point pupil(-1, -1); // Initialiser les coordonnées de la pupille à (-1, -1)

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

            Point currentPupil = findPupil(filtered_eyeROI);
            Point gazeDirection = Point(eyeROI.cols / 2, eyeROI.rows / 2) - currentPupil;

            // Vérifier si la pupille est détectée
            if (currentPupil.x != -1 && currentPupil.y != -1) {
                pupil = currentPupil; // Mettre à jour les coordonnées de la pupille
            }

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

    return {pupil.x, pupil.y, -1}; // Returner les coordonnées de la pupille avec une direction de -1
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


void updateMousePosition(const Point& gazeDirection, int screenWidth, int screenHeight)
{
    const int mouseSpeed = 10; // Réglage de la vitesse de mouvement du curseur

    // Mise à jour des coordonnées X et Y du curseur de la souris en fonction du regard
    mouseX += gazeDirection.x * mouseSpeed;
    mouseY += gazeDirection.y * mouseSpeed;

    // Vérifier si les coordonnées du curseur de la souris sont en dehors de l'écran
    if (mouseX < 0 || mouseX >= screenWidth || mouseY < 0 || mouseY >= screenHeight) {
        mouseX = max(0, min(screenWidth - 1, mouseX)); // Limiter les coordonnées en X
        mouseY = max(0, min(screenHeight - 1, mouseY)); // Limiter les coordonnées en Y
        return; // Sortir de la fonction pour éviter l'OUTPUT [-1, -1, -1]
    }

    // Mettre à jour la position du curseur de la souris en utilisant SDL
    SDL_WarpMouseInWindow(NULL, mouseX, mouseY);
}
