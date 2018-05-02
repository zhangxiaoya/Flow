#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "LKOFlow.hpp"

using namespace cv;
using namespace std;

int main()
{
	if(argc < 3)
	{
		cout << "Usage: " << argv[0] << " : image_name_1 image_name_2" << endl;
		return -1;
	}
	auto img1 = imread(argv[1]);
	if(img1.empty())
	{
		cout << "Open image " << argv[1] << " failed!" << endl;
		return -1;
	}
	auto img2 = imread(argv[2]);
	if(img2.empty())
	{
		cout << "Open image " << argv[2] << " failed!" << endl;
	}

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