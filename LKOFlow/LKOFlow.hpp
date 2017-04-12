#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
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

	static Mat mergeTwoRows(Mat& up, Mat& down);

	static Mat mergeTwoCols(Mat left, Mat right);

	static Mat ResampleImg(Mat& img, Rect& rect, vector<double> disc);

	static void Meshgrid(const Range& xgv, const Range& ygv, Mat& X, Mat& Y);
};

inline vector<double> LKOFlow::PyramidalLKOpticalFlow(Mat& img1, Mat& img2, Rect& ROI)
{
	Mat image1, image2;
	img1.convertTo(image1,CV_32F);
	img2.convertTo(image2,CV_32F);

	auto ROISize = ROI.size();

	auto levels = min(6, static_cast<int>(floor(log2(min(ROISize.height, ROISize.width)) - 2)));

	vector<Mat> image1Pyramid;
	vector<Mat> image2Pyramid;
	image1Pyramid.resize(levels);
	image2Pyramid.resize(levels);

	GaussianPyramid(image1, image1Pyramid, levels);
	GaussianPyramid(image2, image2Pyramid, levels);

	vector<double> disc = {0.0,0.0};

	for (auto curLevel = levels - 1; curLevel >= 0; --curLevel)
	{
		disc[0] *= 2;
		disc[1] *= 2;

		auto scale = pow(2, curLevel);

		Point topLeft;
		topLeft.x = max(static_cast<int>(ceil(ROI.x / scale)), 1);
		topLeft.y = max(static_cast<int>(ceil(ROI.y / scale)), 1);

		Size curSize;
		curSize.width = floor(ROISize.width / scale);
		curSize.height = floor(ROISize.height / scale);

		Point bottomRight;
		bottomRight.x = min(topLeft.x + curSize.width - 1, image1Pyramid[curLevel].size().width - 1);
		bottomRight.y = min(topLeft.y + curSize.height - 1, image1Pyramid[curLevel].size().height - 1);

		IterativeLKOpticalFlow(image1Pyramid[curLevel], image2Pyramid[curLevel], topLeft, bottomRight, disc);
	}

	return disc;
}

inline void LKOFlow::GaussianPyramid(Mat& img, vector<Mat>& pyramid, int levels)
{
	img.copyTo(pyramid[0]);

	auto scale = 2.0;
	auto srcImg = img;

	for (auto i = 1; i < levels; ++i)
	{
		Mat desImg;
		Size size(srcImg.cols / scale, srcImg.rows / scale);

		pyrDown(srcImg, desImg, size);

		desImg.copyTo(pyramid[i]);
		srcImg = pyramid[i];
	}
}

inline void LKOFlow::IterativeLKOpticalFlow(Mat& img1, Mat& img2, Point topLeft, Point bottomRight, vector<double>& disc)
{
	auto oldDisc = disc;

	auto K = 10;
	//	auto stopThrashold = 0.01;
	Rect ROIRect(topLeft, bottomRight);
	auto img1Rect = img1(ROIRect);

	Mat Ht, G;
	ComputeLKFlowParms(img1, Ht, G);

	auto k = 1;
	while (k < K)
	{
		auto resample_img = ResampleImg(img2, ROIRect, disc);
		Mat It = img1Rect - resample_img;

		auto newIt = It.reshape(0, It.rows * It.cols);

		Mat b = Ht * newIt;

		Mat invertG;
		invert(G, invertG);

		Mat dc = invertG * b;

		disc[0] += dc.at<float>(0, 0);
		disc[1] += dc.at<float>(1, 0);

		k++;
	}
}

inline void LKOFlow::ComputeLKFlowParms(Mat& img, Mat& Ht, Mat& G)
{
	Mat SobelX, SobelY;
	Sobel(img, SobelX, CV_32F, 1, 0);
	Sobel(img, SobelY, CV_32F, 0, 1);

	auto X = SobelX(Rect(1, 1, SobelX.cols-2, SobelX.rows-2));
	auto Y = SobelY(Rect(1, 1, SobelY.cols-2, SobelY.rows-2));

	Mat deepCopyedX,deepCopyedY;
	X.copyTo(deepCopyedX);
	Y.copyTo(deepCopyedY);

	auto reshapedX = deepCopyedX.reshape(0, deepCopyedX.rows * deepCopyedX.cols);
	auto reshapedY = deepCopyedY.reshape(0, deepCopyedY.rows * deepCopyedY.cols);

	auto H = mergeTwoCols(reshapedX, reshapedY);
	Ht = H.t();

	G = Ht * H;
}

inline Mat LKOFlow::mergeTwoRows(Mat& up, Mat& down)
{
	auto totalRows = up.rows + down.rows;

	Mat mergedMat(totalRows, up.cols, up.type());

	auto submat = mergedMat.rowRange(0, up.rows);
	up.copyTo(submat);
	submat = mergedMat.rowRange(up.rows, totalRows);
	down.copyTo(submat);

	return mergedMat;
}

inline Mat LKOFlow::mergeTwoCols(Mat left, Mat right)
{
	auto totalCols = left.cols + right.cols;

	Mat mergedDescriptors(left.rows, totalCols, left.type());

	auto submat = mergedDescriptors.colRange(0, left.cols);
	left.copyTo(submat);
	submat = mergedDescriptors.colRange(left.cols, totalCols);
	right.copyTo(submat);

	return mergedDescriptors;
}

inline Mat LKOFlow::ResampleImg(Mat& img, Rect& rect, vector<double> disc)
{
	Mat X, Y;
	auto leftTop = rect.tl();
	auto bottomeRight = rect.br();

	Meshgrid(Range(leftTop.x, bottomeRight.x - 1) - disc[0], Range(leftTop.y, bottomeRight.y - 1) - disc[1], X, Y);

	Mat formatX, formatY;
	X.convertTo(formatX, CV_32FC1);
	Y.convertTo(formatY, CV_32FC1);

	Mat result;
	remap(img, result, formatX, formatY, INTER_LINEAR);

	return result;
}

inline void LKOFlow::Meshgrid(const Range& xgv, const Range& ygv, Mat& X, Mat& Y)
{
	vector<int> t_x, t_y;

	for (auto i = xgv.start; i <= xgv.end; i++)
		t_x.push_back(i);
	for (auto j = ygv.start; j <= ygv.end; j++)
		t_y.push_back(j);

	cv::repeat(cv::Mat(t_x).t(), t_y.size(), 1, X);
	cv::repeat(cv::Mat(t_y), 1, t_x.size(), Y);
}
