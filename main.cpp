#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "BilateralFilter.h"
using namespace cv;

/*
* 程序主入口
*/
int main() {
	//打开源图片
	Mat src = imread("sungou.png"),dst;
	if (src.empty()) {
		std::cout << "图片加载异常" << std::endl;
		return -1;
	}
	//生成双边滤波器对象
	BilateralFilter bilateralfiter;
	bilateralfiter.bilateralfiter(src, dst, Size(KERNEL_SIZE, KERNEL_SIZE), SIGMA_DISTANCE, SIGMA_COLOR);
	//显示滤波之后的图片
	namedWindow("双边滤波处理的图片");
	imshow("双边滤波处理的图片", dst);
	imwrite("sungou_BF.jpg", dst);
	waitKey(0);
	return 0;
}