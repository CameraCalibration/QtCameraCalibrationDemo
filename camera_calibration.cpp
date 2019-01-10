#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>
#include <vector>
#include <stack>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "metrics.h"
#include "geometria.h"

// Parametros del thresholding
#define DIV_S   12
#define Tr       0.15f

// Parametros de contornos
#define MIN_SIZE_CONTOUR    6
#define MYEPS 1e-8

// Constantes usadas para los anillos
#define R_PAR_MIN_ASPECT_RATIO      0.5     // Factor aspect ratio minimo del anillo padre
#define R_CHD_MIN_ASPECT_RATIO      0.4
#define R_CUR_MIN_ASPECT_RATIO      0.55
#define R_PAR_MIN_RECTAN    0.7     // Factor de rectangularidad
#define R_CHD_MIN_RECTAN    0.4
#define R_CUR_MIN_RECTAN    0.75
#define R_CUR_MIN_AREA      0.1

using namespace cv;
using namespace std;

static void help()
{
    cout <<  "This is a camera calibration sample." << endl
         <<  "Usage: camera_calibration [configuration_file -- default ./default.xml]"  << endl
         <<  "Near the sample file you'll find the configuration file, which has detailed help of "
             "how to edit it.  It may be any OpenCV supported file format XML/YAML." << endl;
}
class Settings
{
public:
    Settings() : goodInput(false) {}
    enum Pattern { NOT_EXISTING, CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID, RINGS_GRID };
    enum InputType { INVALID, CAMERA, VIDEO_FILE, IMAGE_LIST };

    void write(FileStorage& fs) const                        //Write serialization for this class
    {
        fs << "{"
                  << "BoardSize_Width"  << boardSize.width
                  << "BoardSize_Height" << boardSize.height
                  << "Square_Size"         << squareSize
                  << "Calibrate_Pattern" << patternToUse
                  << "Calibrate_NrOfFrameToUse" << nrFrames
                  << "Calibrate_FixAspectRatio" << aspectRatio
                  << "Calibrate_AssumeZeroTangentialDistortion" << calibZeroTangentDist
                  << "Calibrate_FixPrincipalPointAtTheCenter" << calibFixPrincipalPoint

                  << "Write_DetectedFeaturePoints" << writePoints
                  << "Write_extrinsicParameters"   << writeExtrinsics
                  << "Write_outputFileName"  << outputFileName

                  << "Show_UndistortedImage" << showUndistorsed

                  << "Input_FlipAroundHorizontalAxis" << flipVertical
                  << "Input_Delay" << delay
                  << "Input" << input
           << "}";
    }
    void read(const FileNode& node)                          //Read serialization for this class
    {
        node["BoardSize_Width" ] >> boardSize.width;
        node["BoardSize_Height"] >> boardSize.height;
        node["Calibrate_Pattern"] >> patternToUse;
        node["Square_Size"]  >> squareSize;
        node["Calibrate_NrOfFrameToUse"] >> nrFrames;
        node["Calibrate_FixAspectRatio"] >> aspectRatio;
        node["Write_DetectedFeaturePoints"] >> writePoints;
        node["Write_extrinsicParameters"] >> writeExtrinsics;
        node["Write_outputFileName"] >> outputFileName;
        node["Calibrate_AssumeZeroTangentialDistortion"] >> calibZeroTangentDist;
        node["Calibrate_FixPrincipalPointAtTheCenter"] >> calibFixPrincipalPoint;
        node["Calibrate_UseFisheyeModel"] >> useFisheye;
        node["Input_FlipAroundHorizontalAxis"] >> flipVertical;
        node["Show_UndistortedImage"] >> showUndistorsed;
        node["Input"] >> input;
        node["Input_Delay"] >> delay;
        node["Fix_K1"] >> fixK1;
        node["Fix_K2"] >> fixK2;
        node["Fix_K3"] >> fixK3;
        node["Fix_K4"] >> fixK4;
        node["Fix_K5"] >> fixK5;

        validate();
    }
    void validate()
    {
        goodInput = true;
        if (boardSize.width <= 0 || boardSize.height <= 0)
        {
            cerr << "Invalid Board size: " << boardSize.width << " " << boardSize.height << endl;
            goodInput = false;
        }
        if (squareSize <= 10e-6)
        {
            cerr << "Invalid square size " << squareSize << endl;
            goodInput = false;
        }
        if (nrFrames <= 0)
        {
            cerr << "Invalid number of frames " << nrFrames << endl;
            goodInput = false;
        }

        if (input.empty())      // Check for valid input
                inputType = INVALID;
        else
        {
            if (input[0] >= '0' && input[0] <= '9')
            {
                stringstream ss(input);
                ss >> cameraID;
                inputType = CAMERA;
            }
            else
            {
                if (isListOfImages(input) && readStringList(input, imageList))
                {
                    inputType = IMAGE_LIST;
                    nrFrames = (nrFrames < (int)imageList.size()) ? nrFrames : (int)imageList.size();
                }
                else
                    inputType = VIDEO_FILE;
            }
            if (inputType == CAMERA)
                inputCapture.open(cameraID);
            if (inputType == VIDEO_FILE)
                inputCapture.open(input);
            if (inputType != IMAGE_LIST && !inputCapture.isOpened())
                    inputType = INVALID;
        }
        if (inputType == INVALID)
        {
            cerr << " Input does not exist: " << input;
            goodInput = false;
        }

        flag = 0;
        if(calibFixPrincipalPoint) flag |= CALIB_FIX_PRINCIPAL_POINT;
        if(calibZeroTangentDist)   flag |= CALIB_ZERO_TANGENT_DIST;
        if(aspectRatio)            flag |= CALIB_FIX_ASPECT_RATIO;
        if(fixK1)                  flag |= CALIB_FIX_K1;
        if(fixK2)                  flag |= CALIB_FIX_K2;
        if(fixK3)                  flag |= CALIB_FIX_K3;
        if(fixK4)                  flag |= CALIB_FIX_K4;
        if(fixK5)                  flag |= CALIB_FIX_K5;

        if (useFisheye) {
            // the fisheye model has its own enum, so overwrite the flags
            flag = fisheye::CALIB_FIX_SKEW | fisheye::CALIB_RECOMPUTE_EXTRINSIC;
            if(fixK1)                   flag |= fisheye::CALIB_FIX_K1;
            if(fixK2)                   flag |= fisheye::CALIB_FIX_K2;
            if(fixK3)                   flag |= fisheye::CALIB_FIX_K3;
            if(fixK4)                   flag |= fisheye::CALIB_FIX_K4;
            if (calibFixPrincipalPoint) flag |= fisheye::CALIB_FIX_PRINCIPAL_POINT;
        }

        calibrationPattern = NOT_EXISTING;
        if (!patternToUse.compare("CHESSBOARD")) calibrationPattern = CHESSBOARD;
        if (!patternToUse.compare("CIRCLES_GRID")) calibrationPattern = CIRCLES_GRID;
        if (!patternToUse.compare("ASYMMETRIC_CIRCLES_GRID")) calibrationPattern = ASYMMETRIC_CIRCLES_GRID;
        if (!patternToUse.compare("RINGS_GRID")) calibrationPattern = RINGS_GRID;
        if (calibrationPattern == NOT_EXISTING)
        {
            cerr << " Camera calibration mode does not exist: " << patternToUse << endl;
            goodInput = false;
        }
        atImageList = 0;

    }
    Mat nextImage()
    {
        Mat result;
        if( inputCapture.isOpened() )
        {
            Mat view0;
            inputCapture >> view0;
            view0.copyTo(result);
        }
        else if( atImageList < imageList.size() )
            result = imread(imageList[atImageList++], IMREAD_COLOR);

        return result;
    }

    static bool readStringList( const string& filename, vector<string>& l )
    {
        l.clear();
        FileStorage fs(filename, FileStorage::READ);
        if( !fs.isOpened() )
            return false;
        FileNode n = fs.getFirstTopLevelNode();
        if( n.type() != FileNode::SEQ )
            return false;
        FileNodeIterator it = n.begin(), it_end = n.end();
        for( ; it != it_end; ++it )
            l.push_back((string)*it);
        return true;
    }

    static bool isListOfImages( const string& filename)
    {
        string s(filename);
        // Look for file extension
        if( s.find(".xml") == string::npos && s.find(".yaml") == string::npos && s.find(".yml") == string::npos )
            return false;
        else
            return true;
    }
public:
    Size boardSize;              // The size of the board -> Number of items by width and height
    Pattern calibrationPattern;  // One of the Chessboard, circles, or asymmetric circle pattern
    float squareSize;            // The size of a square in your defined unit (point, millimeter,etc).
    int nrFrames;                // The number of frames to use from the input for calibration
    float aspectRatio;           // The aspect ratio
    int delay;                   // In case of a video input
    bool writePoints;            // Write detected feature points
    bool writeExtrinsics;        // Write extrinsic parameters
    bool calibZeroTangentDist;   // Assume zero tangential distortion
    bool calibFixPrincipalPoint; // Fix the principal point at the center
    bool flipVertical;           // Flip the captured images around the horizontal axis
    string outputFileName;       // The name of the file where to write
    bool showUndistorsed;        // Show undistorted images after calibration
    string input;                // The input ->
    bool useFisheye;             // use fisheye camera model for calibration
    bool fixK1;                  // fix K1 distortion coefficient
    bool fixK2;                  // fix K2 distortion coefficient
    bool fixK3;                  // fix K3 distortion coefficient
    bool fixK4;                  // fix K4 distortion coefficient
    bool fixK5;                  // fix K5 distortion coefficient

    int cameraID;
    vector<string> imageList;
    size_t atImageList;
    VideoCapture inputCapture;
    InputType inputType;
    bool goodInput;
    int flag;

private:
    string patternToUse;


};

static inline void read(const FileNode& node, Settings& x, const Settings& default_value = Settings())
{
    if(node.empty())
        x = default_value;
    else
        x.read(node);
}

enum { DETECTION = 0, CAPTURING = 1, CALIBRATED = 2 };

bool runCalibrationAndSave(Settings& s, Size imageSize, Mat&  cameraMatrix, Mat& distCoeffs,
                           vector<vector<Point2f> > imagePoints );

bool findRingsGrid(cv::InputArray image, cv::Size patternSize, vector<Point2f> &centers );

int main(int argc, char* argv[])
{
    help();

    //! [file_read]
    Settings s;
    const string inputSettingsFile = argc > 1 ? argv[1] : "D:/opt/windows/Microsoft/VisualStudio/repos/CameraCalibration/QtCameraCalibrationOpenCV/config.xml";
    FileStorage fs(inputSettingsFile, FileStorage::READ); // Read the settings
    if (!fs.isOpened())
    {
        cout << "Could not open the configuration file: \"" << inputSettingsFile << "\"" << endl;
        return -1;
    }
    fs["Settings"] >> s;
    fs.release();                                         // close Settings file
    //! [file_read]

    //FileStorage fout("settings.yml", FileStorage::WRITE); // write config as YAML
    //fout << "Settings" << s;

    if (!s.goodInput)
    {
        cout << "Invalid input detected. Application stopping. " << endl;
        return -1;
    }

    vector<vector<Point2f> > imagePoints;
    Mat cameraMatrix, distCoeffs;
    Size imageSize;
    int mode = s.inputType == Settings::IMAGE_LIST ? CAPTURING : DETECTION;
    clock_t prevTimestamp = 0;
    const Scalar RED(0,0,255), GREEN(0,255,0);
    const char ESC_KEY = 27;

    //! [get_input]
    for(;;)
    {
        Mat view;
        bool blinkOutput = false;

        view = s.nextImage();

        //-----  If no more image, or got enough, then stop calibration and show result -------------
        if( mode == CAPTURING && imagePoints.size() >= (size_t)s.nrFrames )
        {
          if( runCalibrationAndSave(s, imageSize,  cameraMatrix, distCoeffs, imagePoints))
              mode = CALIBRATED;
          else
              mode = DETECTION;
        }
        if(view.empty())          // If there are no more images stop the loop
        {
            // if calibration threshold was not reached yet, calibrate now
            if( mode != CALIBRATED && !imagePoints.empty() )
                runCalibrationAndSave(s, imageSize,  cameraMatrix, distCoeffs, imagePoints);
            break;
        }
        //! [get_input]

        imageSize = view.size();  // Format input image.
        if( s.flipVertical )    flip( view, view, 0 );

        //! [find_pattern]
        vector<Point2f> pointBuf;

        bool found;

        int chessBoardFlags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE;

        if(!s.useFisheye) {
            // fast check erroneously fails with high distortions like fisheye
            chessBoardFlags |= CALIB_CB_FAST_CHECK;
        }

        switch( s.calibrationPattern ) // Find feature points on the input format
        {
        case Settings::CHESSBOARD:
            found = findChessboardCorners( view, s.boardSize, pointBuf, chessBoardFlags);
            break;
        case Settings::CIRCLES_GRID:
            found = findCirclesGrid( view, s.boardSize, pointBuf );
            break;
        case Settings::ASYMMETRIC_CIRCLES_GRID:
            found = findCirclesGrid( view, s.boardSize, pointBuf, CALIB_CB_ASYMMETRIC_GRID );
            break;
        case Settings::RINGS_GRID:
            found = findRingsGrid( view, s.boardSize, pointBuf );
            break;
        default:
            found = false;
            break;
        }
        //! [find_pattern]
        //! [pattern_found]
        if ( found)                // If done with success,
        {
              // improve the found corners' coordinate accuracy for chessboard
                if( s.calibrationPattern == Settings::CHESSBOARD)
                {
                    Mat viewGray;
                    cvtColor(view, viewGray, COLOR_BGR2GRAY);
                    cornerSubPix( viewGray, pointBuf, Size(11,11),
                        Size(-1,-1), TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 30, 0.1 ));
                }

                if( mode == CAPTURING &&  // For camera only take new samples after delay time
                    (!s.inputCapture.isOpened() || clock() - prevTimestamp > s.delay*1e-3*CLOCKS_PER_SEC) )
                {
                    imagePoints.push_back(pointBuf);
                    prevTimestamp = clock();
                    blinkOutput = s.inputCapture.isOpened();
                }

                // Draw the corners.
                drawChessboardCorners( view, s.boardSize, Mat(pointBuf), found );
        }
        //! [pattern_found]
        //----------------------------- Output Text ------------------------------------------------
        //! [output_text]
        string msg = (mode == CAPTURING) ? "100/100" :
                      mode == CALIBRATED ? "Calibrated" : "Press 'g' to start";
        int baseLine = 0;
        Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
        Point textOrigin(view.cols - 2*textSize.width - 10, view.rows - 2*baseLine - 10);

        if( mode == CAPTURING )
        {
            if(s.showUndistorsed)
                msg = format( "%d/%d Undist", (int)imagePoints.size(), s.nrFrames );
            else
                msg = format( "%d/%d", (int)imagePoints.size(), s.nrFrames );
        }

        putText( view, msg, textOrigin, 1, 1, mode == CALIBRATED ?  GREEN : RED);

        if( blinkOutput )
            bitwise_not(view, view);
        //! [output_text]
        //------------------------- Video capture  output  undistorted ------------------------------
        //! [output_undistorted]
        if( mode == CALIBRATED && s.showUndistorsed )
        {
            Mat temp = view.clone();
            if (s.useFisheye)
              cv::fisheye::undistortImage(temp, view, cameraMatrix, distCoeffs);
            else
              undistort(temp, view, cameraMatrix, distCoeffs);
        }
        //! [output_undistorted]
        //------------------------------ Show image and check for input commands -------------------
        //! [await_input]
        imshow("Image View", view);
        char key = (char)waitKey(s.inputCapture.isOpened() ? 50 : s.delay);

        if( key  == ESC_KEY )
            break;

        if( key == 'u' && mode == CALIBRATED )
           s.showUndistorsed = !s.showUndistorsed;

        if( s.inputCapture.isOpened() && key == 'g' )
        {
            mode = CAPTURING;
            imagePoints.clear();
        }
        //! [await_input]
    }

    // -----------------------Show the undistorted image for the image list ------------------------
    //! [show_results]
    if( s.inputType == Settings::IMAGE_LIST && s.showUndistorsed )
    {
        Mat view, rview, map1, map2;

        if (s.useFisheye)
        {
            Mat newCamMat;
            fisheye::estimateNewCameraMatrixForUndistortRectify(cameraMatrix, distCoeffs, imageSize,
                                                                Matx33d::eye(), newCamMat, 1);
            fisheye::initUndistortRectifyMap(cameraMatrix, distCoeffs, Matx33d::eye(), newCamMat, imageSize,
                                             CV_16SC2, map1, map2);
        }
        else
        {
            initUndistortRectifyMap(
                cameraMatrix, distCoeffs, Mat(),
                getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0), imageSize,
                CV_16SC2, map1, map2);
        }

        for(size_t i = 0; i < s.imageList.size(); i++ )
        {
            view = imread(s.imageList[i], IMREAD_COLOR);
            if(view.empty())
                continue;
            remap(view, rview, map1, map2, INTER_LINEAR);
            imshow("Image View", rview);
            char c = (char)waitKey();
            if( c  == ESC_KEY || c == 'q' || c == 'Q' )
                break;
        }
    }
    //! [show_results]

    return 0;
}

///
/// \brief Funcion que aplica Thresholding Adaptativo, este algo está basado en el paper
/// "Adaptive Thresholding Using the Integral Image" de Derek Bradley y Gerhard Roth
/// \param input    Imagen en escala de grises que será segmentada
/// \return         Imagen binaria resultante
///
cv::Mat adaptiveThresholdIntegralImage(cv::Mat input)
{
    // Ancho y altura de la imagen
    int w = input.cols;
    int h = input.rows;
    // Tamaño de la ventana S = (w/DIV_S)
    int s2 = (w / DIV_S) / 2;
    // Declaracion de variables auxiliares
    int sum = 0;
    int count = 0;
    int x1, x2, y1, y2;

    // Imagen integral
    unsigned long *intImg;
    intImg = (unsigned long *)malloc(sizeof(unsigned long)*h*w);

    // Imagen binaria de salida
    cv::Mat binImg(h, w, CV_8UC1);

    // Calculo de la imagen integral basado en los valores de los pixeles de input
    for (int i = 0; i < h; i++) {
        sum = 0;
        for(int j = 0; j < w; j++) {
            sum += input.at<uchar>(i,j);
            if (i == 0)
                intImg[i*w + j] = sum;
            else
                intImg[i*w + j] = intImg[(i-1)*w + j] + sum;
        }
    }

    // Se aplica thresholding y se obtiene la imagen binaria
    for (int i = 0; i < h; i++) {
        for(int j = 0; j < w; j++) {
            // Valores (x1,y1) y (x2,y2) de la ventana SxS
            x1 = j - s2;
            x2 = j + s2;
            y1 = i - s2;
            y2 = i + s2;

            // Verificación de bordes
            if(x1 < 0) x1 = 0;
            if(x2 >= w) x2 = w - 1;
            if(y1 < 0) y1 = 0;
            if(y2 >= h) y2 = h - 1;

            count = (x2 - x1) * (y2 - y1);
            sum = intImg[y2*w + x2] - intImg[y1*w + x2] - intImg[y2*w + x1] + intImg[y1*w + x1];

            // Proceso de binarización
            if((input.at<uchar>(i,j) * count) <= (sum * (1.0 - Tr)))
                binImg.at<uchar>(i,j) = 0;
            else
                binImg.at<uchar>(i,j) = 255;
        }
    }

    free(intImg);
    return binImg;
}

///
/// \brief PatternDetector::cleanNoiseCenters  se encarga de eliminar ruido haciendo analisis de los radios de los contornos hallados
/// \param vCenters Vector de centros de los contornos
/// \param vRadius  Vector de radios de los contornos
/// \return Vector de centros con una reduccion de ruido
///
std::vector<cv::Point2f> cleanNoiseCenters(std::vector<cv::Point2f> vCenters, std::vector<std::pair<float, int> > vRadius, int maxError, cv::Size patternSize)
{
    double radioOptimo;
    // Si el numero de centros es el mismo numero de componentes del patron, se regresa el mismo vector
    if(vCenters.size() <= (patternSize.width * patternSize.height + maxError)) {
        // Ordenamiento de los radios en orden descendente para contar las frecuencias por intervalo
        if(vRadius.size()<=2)
            return vCenters;
        sort(vRadius.rbegin(), vRadius.rend());
        radioOptimo = (vRadius[0].first + vRadius[vRadius.size()-1].first) * 0.5;
        return vCenters;
    }

    std::vector<std::pair<int, float > > freqs;
    std::vector<std::pair<std::pair<float, float>, std::pair<int, int> > > extraInfo;
    float avgVal, stdVal;
    int modeVal, posMode;

    // Se obtiene las frecuencias de los datos agrupados

    metrics::getFrequences<float,int>(vRadius, freqs, extraInfo, false);

    // Se obtiene el promedio y la desviacion estandar
    metrics::getAvgStd(freqs, avgVal, stdVal);

    // Se obtiene la moda y el intervalo en que se encuentra
    metrics::getMode(freqs, modeVal, posMode);

    // Vector donde se almacenaran los centros del patron
    std::vector<cv::Point2f> keypoints;
    float minRad, maxRad;

    // Minimo y maximo radio que debe tener un circulo del patron
    if(modeVal == (int)(patternSize.width * patternSize.height)) {
        minRad = extraInfo[posMode].first.first - ERRORF;
        maxRad = extraInfo[posMode].first.second + ERRORF;
    }
    else {
        minRad = freqs[posMode].second - stdVal;
        maxRad = freqs[posMode].second + stdVal;
    }
    for(size_t i = 0; i < vRadius.size(); i++) {
        if(vRadius[i].first >= minRad && vRadius[i].first <= maxRad) {
            keypoints.push_back(vCenters[vRadius[i].second]);
        }
    }
    // El radio optimo es el label del intervalo moda
    radioOptimo = freqs[posMode].second;
    return keypoints;
}

///
/// \brief PatternDetector::findGrid    Encuentra los contornos de interes
/// \param image    Imagen binaria de entrada
/// \return Vector de centros de los contornos de interes
///
std::vector<cv::Point2f> findGrid(cv::Mat image, cv::Size patternSize)
{
    // Obtencion de los contornos con jerarquia
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(image, contours, hierarchy ,CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

    // Vector de centros
    std::vector<cv::Point2f> keypoints;

    // Variables auxiliares
    double areaPar, auxFactorPar, auxFactorCurr;
    int parent, child;
    cv::Point2f centerCurr, centerPar;

    std::vector<std::pair<float, int> > vectRadios;
    for(size_t i = 0; i < contours.size(); i++)
    {
        parent = hierarchy[i][3];
        child = hierarchy[i][2];

        if(child == -1) {
            if(parent != -1 && hierarchy[i][0] == -1 && hierarchy[i][1] == -1) {
                // PADRE: Rectangulo donde encaja la elipse o el contorno del padre
                cv::RotatedRect boxPar;
                if(contours[parent].size() < MIN_SIZE_CONTOUR)
                    boxPar = minAreaRect(contours[parent]);
                else {
                    cv::Mat pointsf;
                    cv::Mat(contours[parent]).convertTo(pointsf, CV_32F);
                    boxPar = fitEllipse(pointsf);
                }
                centerPar = boxPar.center;

                // ACTUAL: Rectangulo donde encaja la elipse o el contorno del actual
                cv::RotatedRect boxCurr;
                if(contours[i].size() < MIN_SIZE_CONTOUR)
                    centerCurr = centerPar;
                else {
                    cv::Mat pointsf;
                    cv::Mat(contours[i]).convertTo(pointsf, CV_32F);
                    boxCurr = fitEllipse(pointsf);
                    centerCurr = boxCurr.center;
                }

                // Calculo de areas
                areaPar = contourArea(contours[parent]);

                // Factor de aspect ratio
                auxFactorPar = std::min(boxPar.size.width, boxPar.size.height) / std::max(boxPar.size.width, boxPar.size.height);
                auxFactorCurr = std::min(boxCurr.size.width, boxCurr.size.height) / std::max(boxCurr.size.width, boxCurr.size.height);
                if(auxFactorPar < R_PAR_MIN_ASPECT_RATIO || auxFactorCurr < R_CHD_MIN_ASPECT_RATIO)
                    continue;

                // Factor de rectangularidad
                auxFactorPar = areaPar / boxPar.size.area();
                if (auxFactorPar < R_PAR_MIN_RECTAN)
                    continue;

                // Almacenamiento del centro de los anillos concentricos
                keypoints.push_back(cv::Point2f((centerCurr.x + centerPar.x) * 0.5, (centerCurr.y + centerPar.y) * 0.5));
                vectRadios.push_back(std::make_pair(std::max(boxPar.size.width, boxPar.size.height) * 0.5, keypoints.size() - 1));

                // Grafica de los contornos (padre e hijo)
                //drawContours(imgOut, contours, i, cv::Scalar::all(255), 1, 8);
                //drawContours(imgOut, contours, parent, cv::Scalar::all(255), 1, 8);
                // Grafica de las elipses
                //ellipse(imgOut, boxPar, cv::Scalar(0,0,255), 1, CV_AA);
                //ellipse(imgOut, boxCurr, cv::Scalar(0,0,255), 1, CV_AA);
            }
        }
    }
    keypoints = cleanNoiseCenters(keypoints, vectRadios, 2, patternSize);
    // Grafica de los centros despues de la limpieza
    /*for(size_t i = 0; i < keypoints.size(); i++) {
        circle(imgOut, keypoints[i], 3, cv::Scalar(255,255,0), -1);
    }*/
    return keypoints;
}

bool  convexHullCorners(std::vector<cv::Point2f> &keypoints, std::vector<cv::Point2f> &corners)
{
    corners.clear();
    std::vector<std::vector<cv::Point2f> > hull(1);
    cv::convexHull(cv::Mat(keypoints), hull[0], false);

    //Obteniendo las esquinas del convexhull en el patron
    std::vector<int> posCornes = getPosCornes(hull);

    for(int i = 0; i < (int)posCornes.size(); i++){
        corners.push_back(hull[0][posCornes[i]]);
    }

    return true;
}

// Comparador para ordenar los puntos del patron int
bool cmp(std::pair<int,int> p1, std::pair<int,int> p2){
    if(p1.first!=p2.first)
        return p1.first<p2.first;
    return p1.second > p2.second;
}

bool cmpByX(std::pair<int,int> p1, std::pair<int,int> p2){
    if(p1.first!=p2.first)
        return p1.second<p2.second;
    return p1.first > p2.first;
}

// Comparador para hallar los puntos en una recta
bool cmpByDist(std::pair<float,cv::Point2f>  p1, std::pair<float,cv::Point2f> p2){
    return p1.first < p2.first;
}



bool trackingRingsPoints(std::vector<cv::Point2f> &keypoints, std::vector<cv::Point2f> &corners, cv::Size patternSize){

    if(keypoints.size() != patternSize.height*patternSize.width) {
        return false;
    }

    std::vector<std::pair<cv::Point2f,cv::Point2f> > extremosUpDown;
    // Hallando extremos, arriba y abajo
    for(int i = 0; i < corners.size(); i++){
        extremosUpDown.push_back(std::make_pair(corners[i],corners[(i+1) % 4]));
    }

    // Hallando una recta con 6 puntos en su contenido extremosUpDown
    std::vector<std::vector<std::pair<float,float> > > ans;
    for(int i=0;i<(int)extremosUpDown.size();i++){
        cv::Point2f A = extremosUpDown[i].first;
        cv::Point2f B = extremosUpDown[i].second;
        cv::Point2f P;
        // Interseccion de la recta AB con el punto P
        std::vector<std::pair<float,float> > aux;
        for(int k=0;k<(int)keypoints.size();k++){

            // Vemos que no sean los mismo puntos para evitar overflow
            if( (keypoints[k].x == A.x && keypoints[k].y == A.y ) || (keypoints[k].x == B.x && keypoints[k].y == B.y )) continue;
            P = keypoints[k];
            // Hallando la distancia del punto P a la recta AB
            double numerador = (P.x-A.x) * (B.y-A.y) - (P.y-A.y) * (B.x-A.x);
            double denominador = sqrt((B.x - A.x)*(B.x - A.x) + (B.y - A.y)*(B.y - A.y));
            double distancia = numerador / denominador;
            if(abs((int)distancia) < 6){ // se escoge 6 como tolerancia de precision
                aux.push_back(std::make_pair(keypoints[k].x,keypoints[k].y));
            }
        }
        aux.push_back(std::make_pair(A.x,A.y));
        aux.push_back(std::make_pair(B.x,B.y));

        if((int)aux.size()==patternSize.height){
            //Ordenando Ascendentemente x, descendentemente y
            sort(aux.begin(),aux.end(),cmpByX);
            ans.push_back(aux);
        }
    }

    std::vector<std::pair<float,float> > SortPoints;
    std::stack<std::vector<std::pair<float,float> > > pila;
    // escribir lineas de colores
    if(ans.size()>1){

        cv::Point2f PPP = cv::Point2f(ans[0][0].first,ans[0][0].second);
        for(int j=0;j<std::min((int)ans[0].size(),(int)ans[1].size());j++){

            SortPoints.push_back(std::make_pair(ans[0][j].first,ans[0][j].second));
            SortPoints.push_back(std::make_pair(ans[1][j].first,ans[1][j].second));

            std::vector<std::pair<float,cv::Point2f> > distanciaRecta; // Distancia a la recta AB del punto P
            // Hallando los puntos de la recta AB
            cv::Point2f A =  cv::Point2f(ans[0][j].first,ans[0][j].second);
            cv::Point2f B =  cv::Point2f(ans[1][j].first,ans[1][j].second);
            cv::Point2f P;
            // Keypoints tiene todos los puntos del patron
            for(int k=0;k<(int)keypoints.size();k++){
                //Vemos que no sean los mismo puntos para evitar overflow
                if( (keypoints[k].x == A.x && keypoints[k].y == A.y ) || (keypoints[k].x == B.x && keypoints[k].y == B.y )) continue;
                P = keypoints[k];
                // Hallando la distancia del punto P a la recta AB
                double numerador = (P.x-A.x) * (B.y-A.y) - (P.y-A.y) * (B.x-A.x);
                double denominador = sqrt((B.x - A.x)*(B.x - A.x) + (B.y - A.y)*(B.y - A.y));
                double distancia = numerador / denominador;
                distanciaRecta.push_back(std::make_pair(abs((float)distancia),P));
            }

            // Ordenamos las distancias, para escoger los 3 mas cercanos
            std::sort(distanciaRecta.begin(),distanciaRecta.end(),cmpByDist);
            for(int i=0;i<patternSize.width-2;i++){
                SortPoints.push_back(std::make_pair(distanciaRecta[i].second.x,distanciaRecta[i].second.y));
               // circle(img, distanciaRecta[i].second, 5, colors[j%colors.size()], CV_FILLED,8,0);
            }

            //circle(img, Point(ans[1][j].first,ans[1][j].second), 10, CV_RGB(0,0,0), CV_FILLED,8,0);
            std::sort(SortPoints.rbegin(), SortPoints.rend(), [](const std::pair<float, float>& first, const std::pair<float, float>& second){
                return (first.first < second.first);
            });
            // Almacenando los puntos de una recta
            pila.push(SortPoints);
            SortPoints.clear();
        }

        // Escribiendo las rectas de manera descendente
        //int counter = 0;  // Contador para etiquetar los puntos
        keypoints.clear();
        // Extraendo los elementos de la pila
        while(!pila.empty()){
            // Escribiendo numeros
            for(int i=0;i<pila.top().size();i++){
                //std::stringstream sstr;
                //sstr<<counter;
                //counter++;
                //cv::putText(img,sstr.str(),cv::Point2f(pila.top()[i].first,pila.top()[i].second),cv::FONT_HERSHEY_SIMPLEX,0.5,CV_RGB(255,255,255),2);
                keypoints.push_back(cv::Point2f(pila.top()[i].first,pila.top()[i].second));
            }
            pila.pop();
        }
    }

    return true;
}

bool findRingsGrid(cv::InputArray image, cv::Size patternSize, vector<Point2f>& centers ){

    cv::Mat imgGray, imgBlur, imgThresh;

    cv::cvtColor(image, imgGray, CV_BGR2GRAY);
    cv::GaussianBlur(imgGray, imgBlur, cv::Size(3,3), 0.5, 0.5);
    imgThresh = adaptiveThresholdIntegralImage(imgBlur);
    centers = findGrid(imgThresh, patternSize);

    std::vector<cv::Point2f> corners;
    convexHullCorners(centers, corners);

    bool trackCorrect = trackingRingsPoints(centers, corners, patternSize);

    return trackCorrect;
}

//! [compute_errors]
static double computeReprojectionErrors( const vector<vector<Point3f> >& objectPoints,
                                         const vector<vector<Point2f> >& imagePoints,
                                         const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                                         const Mat& cameraMatrix , const Mat& distCoeffs,
                                         vector<float>& perViewErrors, bool fisheye)
{
    vector<Point2f> imagePoints2;
    size_t totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for(size_t i = 0; i < objectPoints.size(); ++i )
    {
        if (fisheye)
        {
            fisheye::projectPoints(objectPoints[i], imagePoints2, rvecs[i], tvecs[i], cameraMatrix,
                                   distCoeffs);
        }
        else
        {
            projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);
        }
        err = norm(imagePoints[i], imagePoints2, NORM_L2);

        size_t n = objectPoints[i].size();
        perViewErrors[i] = (float) std::sqrt(err*err/n);
        totalErr        += err*err;
        totalPoints     += n;
    }

    return std::sqrt(totalErr/totalPoints);
}
//! [compute_errors]
//! [board_corners]
static void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners,
                                     Settings::Pattern patternType /*= Settings::CHESSBOARD*/)
{
    corners.clear();

    switch(patternType)
    {
    case Settings::CHESSBOARD:
    case Settings::CIRCLES_GRID:
    case Settings::RINGS_GRID:
        for( int i = 0; i < boardSize.height; ++i )
            for( int j = 0; j < boardSize.width; ++j )
                corners.push_back(Point3f(j*squareSize, i*squareSize, 0));
        break;

    case Settings::ASYMMETRIC_CIRCLES_GRID:
        for( int i = 0; i < boardSize.height; i++ )
            for( int j = 0; j < boardSize.width; j++ )
                corners.push_back(Point3f((2*j + i % 2)*squareSize, i*squareSize, 0));
        break;
    default:
        break;
    }
}
//! [board_corners]
static bool runCalibration( Settings& s, Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs,
                            vector<vector<Point2f> > imagePoints, vector<Mat>& rvecs, vector<Mat>& tvecs,
                            vector<float>& reprojErrs,  double& totalAvgErr)
{
    //! [fixed_aspect]
    cameraMatrix = Mat::eye(3, 3, CV_64F);
    if( s.flag & CALIB_FIX_ASPECT_RATIO )
        cameraMatrix.at<double>(0,0) = s.aspectRatio;
    //! [fixed_aspect]
    if (s.useFisheye) {
        distCoeffs = Mat::zeros(4, 1, CV_64F);
    } else {
        distCoeffs = Mat::zeros(8, 1, CV_64F);
    }

    vector<vector<Point3f> > objectPoints(1);
    calcBoardCornerPositions(s.boardSize, s.squareSize, objectPoints[0], s.calibrationPattern);

    objectPoints.resize(imagePoints.size(),objectPoints[0]);

    //Find intrinsic and extrinsic camera parameters
    double rms;

    if (s.useFisheye) {
        Mat _rvecs, _tvecs;
        rms = fisheye::calibrate(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, _rvecs,
                                 _tvecs, s.flag);

        rvecs.reserve(_rvecs.rows);
        tvecs.reserve(_tvecs.rows);
        for(int i = 0; i < int(objectPoints.size()); i++){
            rvecs.push_back(_rvecs.row(i));
            tvecs.push_back(_tvecs.row(i));
        }
    } else {
//        rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, s.flag);
        rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);
    }

    cout << "Re-projection error reported by calibrateCamera: "<< rms << endl;

    bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

    totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints, rvecs, tvecs, cameraMatrix,
                                            distCoeffs, reprojErrs, s.useFisheye);

    return ok;
}

// Print camera parameters to the output file
static void saveCameraParams( Settings& s, Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs,
                              const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                              const vector<float>& reprojErrs, const vector<vector<Point2f> >& imagePoints,
                              double totalAvgErr )
{
    FileStorage fs( s.outputFileName, FileStorage::WRITE );

    time_t tm;
    time( &tm );
    struct tm *t2 = localtime( &tm );
    char buf[1024];
    strftime( buf, sizeof(buf), "%c", t2 );

    fs << "calibration_time" << buf;

    if( !rvecs.empty() || !reprojErrs.empty() )
        fs << "nr_of_frames" << (int)std::max(rvecs.size(), reprojErrs.size());
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "board_width" << s.boardSize.width;
    fs << "board_height" << s.boardSize.height;
    fs << "square_size" << s.squareSize;

    if( s.flag & CALIB_FIX_ASPECT_RATIO )
        fs << "fix_aspect_ratio" << s.aspectRatio;

    if (s.flag)
    {
        std::stringstream flagsStringStream;
        if (s.useFisheye)
        {
            flagsStringStream << "flags:"
                << (s.flag & fisheye::CALIB_FIX_SKEW ? " +fix_skew" : "")
                << (s.flag & fisheye::CALIB_FIX_K1 ? " +fix_k1" : "")
                << (s.flag & fisheye::CALIB_FIX_K2 ? " +fix_k2" : "")
                << (s.flag & fisheye::CALIB_FIX_K3 ? " +fix_k3" : "")
                << (s.flag & fisheye::CALIB_FIX_K4 ? " +fix_k4" : "")
                << (s.flag & fisheye::CALIB_RECOMPUTE_EXTRINSIC ? " +recompute_extrinsic" : "");
        }
        else
        {
            flagsStringStream << "flags:"
                << (s.flag & CALIB_USE_INTRINSIC_GUESS ? " +use_intrinsic_guess" : "")
                << (s.flag & CALIB_FIX_ASPECT_RATIO ? " +fix_aspectRatio" : "")
                << (s.flag & CALIB_FIX_PRINCIPAL_POINT ? " +fix_principal_point" : "")
                << (s.flag & CALIB_ZERO_TANGENT_DIST ? " +zero_tangent_dist" : "")
                << (s.flag & CALIB_FIX_K1 ? " +fix_k1" : "")
                << (s.flag & CALIB_FIX_K2 ? " +fix_k2" : "")
                << (s.flag & CALIB_FIX_K3 ? " +fix_k3" : "")
                << (s.flag & CALIB_FIX_K4 ? " +fix_k4" : "")
                << (s.flag & CALIB_FIX_K5 ? " +fix_k5" : "");
        }
        fs.writeComment(flagsStringStream.str());
    }

    fs << "flags" << s.flag;

    fs << "fisheye_model" << s.useFisheye;

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;

    fs << "avg_reprojection_error" << totalAvgErr;
    if (s.writeExtrinsics && !reprojErrs.empty())
        fs << "per_view_reprojection_errors" << Mat(reprojErrs);

    if(s.writeExtrinsics && !rvecs.empty() && !tvecs.empty() )
    {
        CV_Assert(rvecs[0].type() == tvecs[0].type());
        Mat bigmat((int)rvecs.size(), 6, CV_MAKETYPE(rvecs[0].type(), 1));
        bool needReshapeR = rvecs[0].depth() != 1 ? true : false;
        bool needReshapeT = tvecs[0].depth() != 1 ? true : false;

        for( size_t i = 0; i < rvecs.size(); i++ )
        {
            Mat r = bigmat(Range(int(i), int(i+1)), Range(0,3));
            Mat t = bigmat(Range(int(i), int(i+1)), Range(3,6));

            if(needReshapeR)
                rvecs[i].reshape(1, 1).copyTo(r);
            else
            {
                //*.t() is MatExpr (not Mat) so we can use assignment operator
                CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
                r = rvecs[i].t();
            }

            if(needReshapeT)
                tvecs[i].reshape(1, 1).copyTo(t);
            else
            {
                CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
                t = tvecs[i].t();
            }
        }
        fs.writeComment("a set of 6-tuples (rotation vector + translation vector) for each view");
        fs << "extrinsic_parameters" << bigmat;
    }

    if(s.writePoints && !imagePoints.empty() )
    {
        Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
        for( size_t i = 0; i < imagePoints.size(); i++ )
        {
            Mat r = imagePtMat.row(int(i)).reshape(2, imagePtMat.cols);
            Mat imgpti(imagePoints[i]);
            imgpti.copyTo(r);
        }
        fs << "image_points" << imagePtMat;
    }
}

//! [run_and_save]
bool runCalibrationAndSave(Settings& s, Size imageSize, Mat& cameraMatrix, Mat& distCoeffs,
                           vector<vector<Point2f> > imagePoints)
{
    vector<Mat> rvecs, tvecs;
    vector<float> reprojErrs;
    double totalAvgErr = 0;

    bool ok = runCalibration(s, imageSize, cameraMatrix, distCoeffs, imagePoints, rvecs, tvecs, reprojErrs,
                             totalAvgErr);
    cout << (ok ? "Calibration succeeded" : "Calibration failed")
         << ". avg re projection error = " << totalAvgErr << endl;

    if (ok)
        saveCameraParams(s, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, reprojErrs, imagePoints,
                         totalAvgErr);
    return ok;
}
//! [run_and_save]
