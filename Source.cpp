#include <iostream>

#include <vector>
#include <algorithm>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/core/utility.hpp>

#include <thread>

using namespace cv::ml;
using namespace cv;
using namespace std;

//#define STANDARD
#define SINGLE_TEST


int SZ = 20;
float trainPercent = 0.9;
float affineFlags = WARP_INVERSE_MAP | INTER_LINEAR;

Mat HOGToMat(vector<vector<float>> h)
{
    Mat m(h.size(), h[0].size(), CV_32FC1);
    for (int i = 0; i < h.size(); i++)
    {
        for (int j = 0; j < h[0].size(); j++)
        {
            m.at<float>(i, j) = h[i][j];
        }
    }

    return m;
}

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


HOGDescriptor hog(
        Size(20, 20), //winSize
        Size(10, 10), //blocksize
        Size(5, 5), //blockStride,
        Size(10,10), //cellSize,
        9, //nbins,
        1, //derivAper,
        -1, //winSigma,
        HOGDescriptor::L2Hys, //histogramNormType,
        0.2, //L2HysThresh,
        false, //gammal correction,
        64, //nlevels=64
        true); //signedGradient 


int main()
{

    string imgPath = "digits.png";
    Mat image = imread(imgPath, 0);
#ifdef SINGLE_TEST
    string singleTestDigitPath = "Single Tests/3456 test.png"; // Color does not matter!
    Mat singleTestDigit = imread(singleTestDigitPath, 0);
    int xsSZ = singleTestDigit.cols, ysSZ = singleTestDigit.rows;
    double xScale = (double)SZ / xsSZ, yScale = (double)SZ / ysSZ;
    resize(singleTestDigit, singleTestDigit, Size(), xScale, yScale);

    cout << singleTestDigit.rows << endl;
#endif
    //namedWindow("Dataset", WINDOW_FREERATIO);
    //imshow("Dataset", image);

    int dataCount = 0;
    vector<Mat> trainingDigits, testDigits;
    //namedWindow("Data", WINDOW_FREERATIO);
    //namedWindow("Deskewed", WINDOW_FREERATIO);
    //cout << (float)(trainPercent*image.cols) << endl;
    for (int i = 0; i < image.rows; i += SZ)
    {
        for (int j = 0; j < image.cols; j += SZ)
        {
            //cout << i << " " << j << endl;
            Mat currData = image.rowRange(i, i + SZ).colRange(j, j + SZ);
            //imshow("Data", currData);
            currData = deskew(currData);
            //imshow("Deskewed", currData);
            if (j < (float)(trainPercent * image.cols))
            {
                trainingDigits.push_back(currData);
            }
#ifdef STANDARD
            else
            {
                testDigits.push_back(currData);
            }
#endif
            dataCount++;
        }
    }
#ifdef SINGLE_TEST
    testDigits.push_back(singleTestDigit);
#endif
    destroyAllWindows();
    //namedWindow("Data", WINDOW_FREERATIO);
    //cout << "Inputted images" << endl;
    //cout << "dataCount: " << dataCount << endl;
    //cout << "trainingDigits.size(): " << trainingDigits.size() << endl;

    vector<int> trainingLabels, testLabels;
    int d{ 0 };
    for (int i = 0; i < trainPercent * dataCount; i++)
    {
        if (i % ((int)(trainPercent * dataCount / 10)) == 0 and i != 0)
        {
            d++;
        }
        trainingLabels.push_back(d);
    }
#ifdef STANDARD
    d = 0;
    for (int i = 0; i < (int)((1 - trainPercent) * dataCount); i++)
    {
        if (i % ((int)((1 - trainPercent) * dataCount / 10)) == 0 and i != 0)
        {
            d++;
        }
        testLabels.push_back(d);
    }
#endif
#ifdef SINGLE_TEST
    testLabels.push_back(-1);
#endif

    vector<float> descriptors;
    vector<vector<float>> trainingHOG;
    for (int i = 0; i < trainingDigits.size(); i++)
    {
        //cout << "trainingDigts[" << i << "]" << endl;
        //imshow("Data", trainingDigits[i]);
        //cout << "size of trainingDigits[" << i << "]: " << trainingDigits[i].rows << "x" << trainingDigits[i].cols << endl;
        hog.compute(trainingDigits[i], descriptors, Size(4, 4));
        //cout << "  computed hog" << endl;
        trainingHOG.push_back(descriptors);
    }
    vector<vector<float>> testHOG;
#ifdef STANDARD
    for (int i = 0; i < testDigits.size(); i++)
    {
        hog.compute(testDigits[i], descriptors, Size(4, 4));
        testHOG.push_back(descriptors);
    }
#endif
#ifdef SINGLE_TEST
    hog.compute(testDigits[0], descriptors, Size(4, 4));
    testHOG.push_back(descriptors);
#endif
    Mat trainingMat = HOGToMat(trainingHOG);
    Mat testMat = HOGToMat(testHOG);

    /*cout << trainingMat << endl;
    cout << testMat << endl;*/

    cout << "Computed HOGs" << endl;

    Ptr<SVM> svm = SVM::create(); // Remember: SVM is used to find boundaries of where to classify digits
    svm->setGamma(0.50625); // Value for Kernel
    svm->setC(12.5); //Value for Type
    svm->setKernel(SVM::RBF); // Method helping classification
    svm->setType(SVM::C_SVC); // i guess this is what the SVM's job is; do some do different tasks?
    Ptr<TrainData> td = TrainData::create(trainingMat, ROW_SAMPLE, trainingLabels); // takes trainingMat row by row and matches the values in training labels to trainingMat values
    svm->train(td);
    //svm->trainAuto(td); // finds optimal Gamma and C values, takes longer, no need here
    svm->save("digits5000_model.yml"); // Save SVM model


    Mat testResponse;
    svm->predict(testMat, testResponse); // testMat has HOGS of samples, testResponse will be filled with results

    //cout << testResponse << endl;

    namedWindow("Test Data", WINDOW_FREERATIO);
    double nCorrect{ 0 }, accuracy;
    for (int i = 0; i < testResponse.rows; i++)
    {
        imshow("Test Data", testDigits[i]);
        cout << "Response: " << testResponse.at<float>(i, 0) << " Label:" << testLabels[i] << endl;
        nCorrect += testResponse.at<float>(i, 0) == testLabels[i];
        waitKey();
    }
    accuracy = 100 * nCorrect / testLabels.size();
    this_thread::sleep_for(2s); // Don't exit console immediately!

    cout << "Accuracy: " << accuracy << "%" << endl;

    return 0;
}