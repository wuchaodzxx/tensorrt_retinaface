#pragma once

#include <opencv2/core/mat.hpp>


/**
 *
 * Define some image helper functions
 */


typedef struct {
	int w;
	int h;
	int c;
	float *data;
} image;

float* normal(cv::Mat img);
float* HWC2CHW(cv::Mat img, const float kMeans[3]);
cv::Mat read2mat(float * prob, cv::Mat out);
cv::Mat map2threeunchar(cv::Mat real_out, cv::Mat real_out_);
