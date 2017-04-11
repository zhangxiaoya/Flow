#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace cv;
using namespace std;

int main()
{
	Mat image = imread("timg.jpg");
	if (image.data)
	{
		imshow("test Image show", image);
		waitKey(0);
		destroyAllWindows();
	}
}