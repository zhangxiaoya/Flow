#include <opencv2\core/core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <vector>
#include <cmath>

using namespace std;
using namespace cv;

class LKOFlow
{
public:
	static vector<double> PyramidalLKOpticalFlow(Mat& img1, Mat& img2, Rect& ROI);

private:
	static void GaussianPyramid(Mat& img, vector<Mat>& pyramid, int levels);

	static void IterativeLKOpticalFlow(Mat& Pyramid1, Mat& Pyramid2, Point topLeft, Point bottomRight, vector<double>& disc);

	static void ComputeLKFlowParms(Mat& img, Mat& Ht, Mat& G);
	
	static Mat mergeRows(Mat& left, Mat& right);

	static Mat ResampleImg(Mat& img, Rect& rect, vector<double> disc);

	static void Meshgrid(const Range &xgv, const Range &ygv, Mat &X, Mat &Y);
};

vector<double> LKOFlow::PyramidalLKOpticalFlow(Mat& img1, Mat& img2, Rect& ROI)
{
	Mat image1, image2;
	img1.convertTo(image1,CV_32F);
	img2.convertTo(image2,CV_32F);

	Size ROISize =  ROI.size();

	int levels = min(6, (int)floor(log2(min(ROISize.height, ROISize.width)) - 2));

	vector<Mat> image1Pyramid;
	vector<Mat> image2Pyramid;
	image1Pyramid.resize(levels);
	image2Pyramid.resize(levels);

	GaussianPyramid(image1, image1Pyramid, levels);
	GaussianPyramid(image2, image2Pyramid, levels);

	vector<double> disc = { 0.0,0.0 };

	for (int curLevel = levels; curLevel >= 0; --curLevel)
	{
		disc[0] *= 2;
		disc[1] *= 2;

		double scale = pow(2, curLevel - 1);

		Point topLeft;
		topLeft.x = max((int)ceil(ROI.x / scale),2);
		topLeft.y = max((int)ceil(ROI.y / scale),2);

		Size curSize;
		curSize.width = floor(ROISize.width / scale);
		curSize.height = floor(ROISize.height / scale);

		Point bottomRight;
		bottomRight.x = min(topLeft.x + curSize.width - 1, image1Pyramid[curLevel].size().width -1);
		bottomRight.y = min(topLeft.y + curSize.height - 1, image1Pyramid[curLevel].size().height -1);

		IterativeLKOpticalFlow(image1Pyramid[curLevel], image2Pyramid[curLevel], topLeft, bottomRight, disc);
	}

	return disc;
}

void LKOFlow::GaussianPyramid(Mat& img, vector<Mat>& pyramid, int levels)
{
	for (int i = 0; i < levels; ++i)
	{
		double scale = pow(2, i);
		Mat outImage;
		pyrDown(img, outImage, Size(img.cols / scale, img.rows / scale));
		pyramid.push_back(outImage);
	}
}

void LKOFlow::IterativeLKOpticalFlow(Mat& Pyramid1, Mat& Pyramid2, Point topLeft, Point bottomRight, vector<double>& disc)
{
	vector<double> oldDisc = disc;

	int K = 10;
	double stopThrashold = 0.01;

	Rect ROIRect(topLeft,bottomRight);
	//Mat pimg1 = Pyramid1(ROIRect);

	Mat Ht, G;

	ComputeLKFlowParms(Pyramid1, Ht,G);

	int k = 1;
	while (k < K)
	{
		Mat It = Pyramid1 - ResampleImg(Pyramid2, ROIRect, disc);

		It.reshape(0, It.rows* It.cols);

		Mat b = Ht * It;
		
		Mat invertG;
		invert(G, invertG);

		Mat dc = invertG * b;

		disc[0] += dc.at<uchar>(0,0);
		disc[1] += dc.at<uchar>(0,1);

		k++;
	}
}

void LKOFlow::ComputeLKFlowParms(Mat& img, Mat& Ht, Mat& G)
{
	Mat SobelX, SobelY;
	Sobel(img, SobelX, CV_32F, 1, 0);
	Sobel(img, SobelY, CV_32F, 0, 1);

	Mat X = SobelX(Rect(2, 2, SobelX.cols - 2, SobelX.rows - 2));
	Mat Y = SobelY(Rect(2, 2, SobelY.cols - 2, SobelY.rows - 2));

	X.reshape(0,X.rows * X.cols);
	Y.reshape(0,Y.rows * Y.cols);

	Mat H = mergeRows(X, Y);
	Ht = H.t();

	G = Ht*H;
}

Mat LKOFlow::mergeRows(Mat& left, Mat& right)
{
	int totalRows = left.rows + right.rows;

	Mat mergedMat(totalRows, left.cols, left.type());

	Mat submat = mergedMat.rowRange(0, left.rows);
	left.copyTo(submat);
	submat = mergedMat.rowRange(left.rows, totalRows);
	right.copyTo(submat);

	return mergedMat;
}

Mat LKOFlow::ResampleImg(Mat& img, Rect& rect, vector<double> disc)
{
	Mat X, Y;
	Point leftTop = rect.tl();
	Point bottomeRight = rect.br();

	Meshgrid(Range(leftTop.x, bottomeRight.x) - disc[0], Range(leftTop.y, bottomeRight.y) - disc[1], X, Y);
	Mat result;
	remap(img, result, X, Y, INTER_LINEAR);
	return result;
}

void LKOFlow::Meshgrid(const Range &xgv, const Range &ygv, Mat &X, Mat &Y)
{
	vector<int> t_x, t_y;

	for (int i = xgv.start; i <= xgv.end; i++) 
		t_x.push_back(i);
	for (int j = ygv.start; j <= ygv.end; j++) 
		t_y.push_back(j);

	cv::repeat(cv::Mat(t_x).t(), t_y.size(), 1, X);
	cv::repeat(cv::Mat(t_y), 1, t_x.size(), Y);
}
