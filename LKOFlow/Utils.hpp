#pragma once
#include <core/core.hpp>

class Utils
{
public:
	static cv::Mat ReshapedMatColumnFirst(const cv::Mat& srcMat);
};

inline cv::Mat Utils::ReshapedMatColumnFirst(const cv::Mat& srcMat)
{
	cv::Mat reshapedMat(cv::Size(1, srcMat.cols * srcMat.rows), CV_32FC1);

	for (auto r = 0; r < srcMat.rows; ++r)
	{
		auto nr = r;
		auto rowSrcMat = srcMat.ptr<float>(r);
		for (auto c = 0; c < srcMat.cols; ++c)
		{
			reshapedMat.ptr<float>(nr)[0] = rowSrcMat[c];
			nr += srcMat.rows;
		}
	}
	return reshapedMat;
}
