#pragma once
constexpr auto KERNEL_SIZE = 31; //卷积核大小
constexpr auto COLOR_KERNEL_SIZE = 256; //灰度值差异
constexpr auto SIGMA_DISTANCE = 10; //
constexpr auto SIGMA_COLOR = 100;

/*
* 双边滤波器类
*/
class BilateralFilter {
public:
    BilateralFilter() {};
    void getDistanceWeight(cv::Mat& Mask, cv::Size wsize, double spaceSigma);
    void getColorWeight(std::vector<double>& colorMask, double colorSigma);
    void bilateralfiter(cv::Mat& src, cv::Mat& dst, cv::Size wsize, double spaceSigma, double colorSigma);
};