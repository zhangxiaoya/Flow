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
	static void GaussianDownSample(vector<Mat>::const_reference srcMat, vector<Mat>::reference destMat);

	static void GaussianPyramid(Mat& img, vector<Mat>& pyramid, int levels);

	static void ChangeToFloat(Mat& srcImg, Mat& destImg);

	static double MyNorm(const Mat& mat);

	static void IterativeLKOpticalFlow(Mat& Pyramid1, Mat& Pyramid2, Point topLeft, Point bottomRight, vector<double>& disc);

	static void ComputeLKFlowParms(Mat& img, Mat& Ht, Mat& G);

	static Mat MergeTwoRows(Mat& up, Mat& down);

	static Mat MergeTwoCols(Mat left, Mat right);

	static Mat ResampleImg(Mat& img, Rect& rect, vector<double> disc);

	static void Meshgrid(const Range& xgv, const Range& ygv, Mat& X, Mat& Y);
};

inline void LKOFlow::ChangeToFloat(Mat& srcImg, Mat& destImg)
{
	if (srcImg.type() != CV_32FC1)
		srcImg.convertTo(destImg,CV_32FC1);
}

inline vector<double> LKOFlow::PyramidalLKOpticalFlow(Mat& img1, Mat& img2, Rect& ROI)
{
	Mat image1, image2;
	ChangeToFloat(img1, image1);
	ChangeToFloat(img2, image2);

	auto ROISize = ROI.size();

	auto levels = min(6, static_cast<int>(floor(log2(min(ROISize.height, ROISize.width)) - 2)));

	vector<Mat> image1Pyramid;
	vector<Mat> image2Pyramid;
	image1Pyramid.resize(levels);
	image2Pyramid.resize(levels);

	GaussianPyramid(image1, image1Pyramid, levels);
	GaussianPyramid(image2, image2Pyramid, levels);

	vector<double> distance = {0.0,0.0};

	for (auto currentLevel = levels - 1; currentLevel >= 0; --currentLevel)
	{
		distance[0] *= 2;
		distance[1] *= 2;

		auto scale = pow(2, currentLevel);

		Point topLeft;
		topLeft.x = max(static_cast<int>(ceil(ROI.x / scale)), 1);
		topLeft.y = max(static_cast<int>(ceil(ROI.y / scale)), 1);

		Size currentSize;
		currentSize.width = floor(ROISize.width / scale);
		currentSize.height = floor(ROISize.height / scale);

		Point bottomRight;
		bottomRight.x = min(topLeft.x + currentSize.width - 1, image1Pyramid[currentLevel].size().width - 1);
		bottomRight.y = min(topLeft.y + currentSize.height - 1, image1Pyramid[currentLevel].size().height - 1);

		IterativeLKOpticalFlow(image1Pyramid[currentLevel], image2Pyramid[currentLevel], topLeft, bottomRight, distance);
	}

	return distance;
}

inline void LKOFlow::GaussianDownSample(vector<Mat>::const_reference src_mat, vector<Mat>::reference dest_mat)
{
	Mat kernel = (Mat_<float>(1, 5) << 0.0625, 0.2500, 0.3750, 0.2500, 0.0625);
	Mat kernelT = kernel.t();

	Mat img, imgT;
	filter2D(src_mat, img, CV_32F, kernel, Point(-1, -1), 0, BORDER_REFLECT);
	filter2D(img, imgT, CV_32F, kernelT, Point(-1, -1), 0, BORDER_REFLECT);

	Size size(ceil(src_mat.cols / 2.0), ceil(src_mat.rows / 2.0));
	Mat tempImg(size, CV_32FC1);

	for (auto r = 0; r < imgT.rows; r += 2)
	{
		auto rowSrcMat = imgT.ptr<float>(r);
		auto rowDstmat = tempImg.ptr<float>(ceil(r / 2.0));

		for (auto c = 0; c < imgT.cols; c += 2)
		{
			int idx = ceil(c / 2.0);
			rowDstmat[idx] = rowSrcMat[c];
		}
	}

	tempImg.copyTo(dest_mat);
}

inline void LKOFlow::GaussianPyramid(Mat& img, vector<Mat>& pyramid, int levels)
{
	img.copyTo(pyramid[0]);

	for (auto i = 1; i < levels; ++i)
		GaussianDownSample(pyramid[i - 1], pyramid[i]);
}

inline double LKOFlow::MyNorm(const Mat& mat)
{
	/*
	 * special use: Mat is a (2*1) vector, only get norm of this vector
	 */
	double sum = mat.at<float>(0, 0) * mat.at<float>(0, 0) + mat.at<float>(1, 0) * mat.at<float>(1, 0);
	return sqrt(sum);
}

inline void LKOFlow::IterativeLKOpticalFlow(Mat& img1, Mat& img2, Point topLeft, Point bottomRight, vector<double>& distance)
{
	auto oldDistance = distance;

	auto maxIterativeCount = 10;
	auto stopThrashold = 0.01;
	Rect ROIRect(topLeft, bottomRight);
	auto img1Rect = img1(ROIRect);

	Mat Ht, G;
	ComputeLKFlowParms(img1, Ht, G);

	auto currentIterativeIndex = 1;
	double normDistrance = 1;
	while (currentIterativeIndex < maxIterativeCount && normDistrance > stopThrashold)
	{
		auto resample_img = ResampleImg(img2, ROIRect, distance);
		Mat It = img1Rect - resample_img;

		auto newIt = It.reshape(0, It.rows * It.cols);

		Mat b = Ht * newIt;

		Mat invertG;
		invert(G, invertG);

		Mat dc = invertG * b;

		normDistrance = MyNorm(dc);

		distance[0] += dc.at<float>(0, 0);
		distance[1] += dc.at<float>(1, 0);

		currentIterativeIndex++;
	}
}

inline void LKOFlow::ComputeLKFlowParms(Mat& img, Mat& Ht, Mat& G)
{
	Mat SobelX, SobelY;
	Sobel(img, SobelX, CV_32F, 1, 0);
	Sobel(img, SobelY, CV_32F, 0, 1);

	auto rectSobelX = SobelX(Rect(1, 1, SobelX.cols - 2, SobelX.rows - 2));
	auto rectSobelY = SobelY(Rect(1, 1, SobelY.cols - 2, SobelY.rows - 2));

	Mat deepCopyedX, deepCopyedY;
	rectSobelX.copyTo(deepCopyedX);
	rectSobelY.copyTo(deepCopyedY);

	auto reshapedX = deepCopyedX.reshape(0, deepCopyedX.rows * deepCopyedX.cols);
	auto reshapedY = deepCopyedY.reshape(0, deepCopyedY.rows * deepCopyedY.cols);

	auto H = MergeTwoCols(reshapedX, reshapedY);
	Ht = H.t();

	G = Ht * H;
}

inline Mat LKOFlow::MergeTwoRows(Mat& up, Mat& down)
{
	auto totalRows = up.rows + down.rows;

	Mat mergedMat(totalRows, up.cols, up.type());

	auto submat = mergedMat.rowRange(0, up.rows);
	up.copyTo(submat);
	submat = mergedMat.rowRange(up.rows, totalRows);
	down.copyTo(submat);

	return mergedMat;
}

inline Mat LKOFlow::MergeTwoCols(Mat left, Mat right)
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
