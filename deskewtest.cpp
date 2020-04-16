#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/ml.hpp>

using namespace cv;
using namespace std;

int SZ = 20;
float affineFlags = WARP_INVERSE_MAP | INTER_LINEAR;

Mat deskew(Mat& img) {
    Moments m = moments(img);
    if (abs(m.mu02) < 1e-2) {
        return img.clone();
    }
    float skew = m.mu11 / m.mu02;
    Mat warpMat = (Mat_<float>(2, 3) << 1, skew, -0.5 * SZ * skew, 0, 1, 0), 
        imgOut = Mat::zeros(img.rows, img.cols, img.type());
    warpAffine(img, imgOut, warpMat, imgOut.size(), affineFlags);

    return imgOut;
}

int main()
{

    string imgName = "five.png";
    namedWindow("Before", WINDOW_FREERATIO);
    namedWindow("After", WINDOW_FREERATIO);

	Mat image = imread(imgName, 0);

    if (image.empty())
    {
        cout << "The image is empty!" << endl;
    }
    else
    {
        cout << image << endl;

        waitKey(1000);

        imshow("Before", image);
        waitKey(0);

        image = deskew(image);

        cout << "helo?" << endl;

        imshow("After", image);
        waitKey(0);

        return 0;
    }
}