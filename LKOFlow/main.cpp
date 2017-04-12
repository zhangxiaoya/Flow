#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include "LKOFlow.hpp"

using namespace cv;
using namespace std;

int main()
{
	Mat img1 = imread("1.jpg");
	Mat img2 = imread("2.jpg");
	Mat graImg1;
	Mat graImg2;

	if (img1.data)
	{
		imshow("Image1", img1);
		cout << img1.channels() << endl;
		cout << img1.rows << endl;
		cout << img1.cols << endl;
		cvtColor(img1, graImg1, CV_BGR2GRAY);
		cout << graImg1.channels() << endl;
		waitKey(0);
	}

	if (img2.data)
	{
		imshow("Image2", img2);
		cout << img2.channels() << endl;
		cvtColor(img2, graImg2, CV_BGR2GRAY);
		cout << graImg2.channels() << endl;
		waitKey(0);
	}

	destroyAllWindows();
	return 0;
}