#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	namedWindow("Picture", WINDOW_AUTOSIZE);// Create a window for display.
    Mat image;
	string imgName;
	if (argc >= 2)
	{
		for (int i = 1; i < argc; i++)
		{
			cout << i << ". ";
			imgName = argv[i];
			cout << imgName << endl;
			image = imread(imgName);   // Read the file

			if (!image.data)                              // Check for invalid input
			{
				cout << "    Could not open or find the image" << endl;
				return -1;
			}

			cout << ".size(): " << image.size() << endl;
			imshow("Picture", image);                   // Show our image inside it.

			waitKey(0);                                          // Wait for a keystroke in the window
		}
	}
	
	destroyWindow("Picture");
    return 0;
}