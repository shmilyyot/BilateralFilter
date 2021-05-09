#include <math.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "BilateralFilter.h"
using namespace cv;

/*
* 函数定义文件
*/

// 空间域权重确定
void BilateralFilter::getDistanceWeight(Mat& Mask,Size wsize, double spaceSigma) {
	Mask.create(wsize, CV_64F);
	int h = wsize.height;
	int w = wsize.width;
	int center_h = (h - 1) / 2;
	int center_w = (w - 1) / 2;
	double sum = 0.0;
	double x, y;

	for (int i = 0; i < h; ++i) {
		y = pow(i - center_h, 2);
		double* Maskdate = Mask.ptr<double>(i);
		for (int j = 0; j < w; ++j) {
			x = pow(j - center_w, 2);
			double g = exp(-(x + y) / (2 * spaceSigma * spaceSigma));
			Maskdate[j] = g;
			sum += g;
		}
	}
}

// 颜色域权重确定
void BilateralFilter::getColorWeight(std::vector<double>& colorMask, double colorSigma) {
	for (int i = 0; i < COLOR_KERNEL_SIZE; ++i) {
		double colordiff = exp(-(i * i) / (2 * colorSigma * colorSigma));
		colorMask.push_back(colordiff);
	}

}

// 双边滤波
void BilateralFilter::bilateralfiter(Mat& src, Mat& dst, Size wsize, double spaceSigma, double colorSigma) {
	Mat spaceMask;
	std::vector<double> colorMask;
	Mat Mask0 = Mat::zeros(wsize, CV_64F);
	Mat Mask1 = Mat::zeros(wsize, CV_64F);
	Mat Mask2 = Mat::zeros(wsize, CV_64F);
	//空间模板
	getDistanceWeight(spaceMask, wsize, spaceSigma);
	//值域模板
	getColorWeight(colorMask, colorSigma);
	int hh = (wsize.height - 1) / 2;
	int ww = (wsize.width - 1) / 2;
	dst.create(src.size(), src.type());
	//边界填充
	Mat Newsrc;
	copyMakeBorder(src, Newsrc, hh, hh, ww, ww, BORDER_REPLICATE);
	for (int i = hh; i < src.rows + hh; ++i) {
		for (int j = ww; j < src.cols + ww; ++j) {
			double sum[3] = { 0 };
			int graydiff[3] = { 0 };
			double space_color_sum[3] = { 0.0 };
			for (int r = -hh; r <= hh; ++r) {
				for (int c = -ww; c <= ww; ++c) {
					if (src.channels() == 1) {
						int centerPix = Newsrc.at<uchar>(i, j);
						int pix = Newsrc.at<uchar>(i + r, j + c);
						graydiff[0] = abs(pix - centerPix);
						double colorWeight = colorMask[graydiff[0]];
						Mask0.at<double>(r + hh, c + ww) = colorWeight * spaceMask.at<double>(r + hh, c + ww);
						space_color_sum[0] = space_color_sum[0] + Mask0.at<double>(r + hh, c + ww);

					}
					else if (src.channels() == 3) {
						Vec3b centerPix = Newsrc.at<Vec3b>(i, j);
						Vec3b bgr = Newsrc.at<Vec3b>(i + r, j + c);
						graydiff[0] = abs(bgr[0] - centerPix[0]); graydiff[1] = abs(bgr[1] - centerPix[1]); graydiff[2] = abs(bgr[2] - centerPix[2]);
						double colorWeight0 = colorMask[graydiff[0]];
						double colorWeight1 = colorMask[graydiff[1]];
						double colorWeight2 = colorMask[graydiff[2]];
						Mask0.at<double>(r + hh, c + ww) = colorWeight0 * spaceMask.at<double>(r + hh, c + ww);
						Mask1.at<double>(r + hh, c + ww) = colorWeight1 * spaceMask.at<double>(r + hh, c + ww);
						Mask2.at<double>(r + hh, c + ww) = colorWeight2 * spaceMask.at<double>(r + hh, c + ww);
						space_color_sum[0] = space_color_sum[0] + Mask0.at<double>(r + hh, c + ww);
						space_color_sum[1] = space_color_sum[1] + Mask1.at<double>(r + hh, c + ww);
						space_color_sum[2] = space_color_sum[2] + Mask2.at<double>(r + hh, c + ww);
					}
				}
			}

			//归一化
			if (src.channels() == 1)
				Mask0 = Mask0 / space_color_sum[0];
			else {
				Mask0 = Mask0 / space_color_sum[0];
				Mask1 = Mask1 / space_color_sum[1];
				Mask2 = Mask2 / space_color_sum[2];
			}

			for (int r = -hh; r <= hh; ++r) {
				for (int c = -ww; c <= ww; ++c) {

					if (src.channels() == 1) {
						sum[0] = sum[0] + Newsrc.at<uchar>(i + r, j + c) * Mask0.at<double>(r + hh, c + ww); 
					}
					else if (src.channels() == 3) {
						Vec3b bgr = Newsrc.at<Vec3b>(i + r, j + c); 
						sum[0] = sum[0] + bgr[0] * Mask0.at<double>(r + hh, c + ww);
						sum[1] = sum[1] + bgr[1] * Mask1.at<double>(r + hh, c + ww);
						sum[2] = sum[2] + bgr[2] * Mask2.at<double>(r + hh, c + ww);
					}
				}
			}

			for (int k = 0; k < src.channels(); ++k) {
				if (sum[k] < 0)
					sum[k] = 0;
				else if (sum[k] > 255)
					sum[k] = 255;
			}
			if (src.channels() == 1){
				dst.at<uchar>(i - hh, j - ww) = static_cast<uchar>(sum[0]);
			}
			else if (src.channels() == 3){
				Vec3b bgr = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
				dst.at<Vec3b>(i - hh, j - ww) = bgr;
			}

		}
	}

}
