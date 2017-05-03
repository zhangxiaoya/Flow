#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "LKOFlow.hpp"

using namespace cv;
using namespace std;

int main()
{
	auto img1 = imread("1.jpg");
	auto img2 = imread("2.jpg");

	Mat graImg1;
	Mat graImg2;

	if (img1.data)
		cvtColor(img1, graImg1, CV_BGR2GRAY);

	if (img2.data)
		cvtColor(img2, graImg2, CV_BGR2GRAY);

	Rect roi(0, 0, graImg1.cols, graImg1.rows);
	auto disc = LKOFlow::PyramidalLKOpticalFlow(graImg1, graImg2, roi);

	destroyAllWindows();

	cout << "Distances:" << endl;
	cout << "X = " << disc[0] << endl << "Y = " << disc[1] << endl;

	system("pause");
	return 0;
}