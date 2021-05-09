#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "BilateralFilter.h"
using namespace cv;

/*
* ���������
*/
int main() {
	//��ԴͼƬ
	Mat src = imread("sungou.png"),dst;
	if (src.empty()) {
		std::cout << "ͼƬ�����쳣" << std::endl;
		return -1;
	}
	//����˫���˲�������
	BilateralFilter bilateralfiter;
	bilateralfiter.bilateralfiter(src, dst, Size(KERNEL_SIZE, KERNEL_SIZE), SIGMA_DISTANCE, SIGMA_COLOR);
	//��ʾ�˲�֮���ͼƬ
	namedWindow("˫���˲������ͼƬ");
	imshow("˫���˲������ͼƬ", dst);
	imwrite("sungou_BF.jpg", dst);
	waitKey(0);
	return 0;
}