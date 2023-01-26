//
// Created by timber3252 on 2023/1/26.
//

#ifndef ESLAB_UTIL_IMAGE_HPP
#define ESLAB_UTIL_IMAGE_HPP

#include <opencv2/opencv.hpp>

cv::Mat image_resize(const cv::Mat &image,  std::int32_t width, std::int32_t height);
cv::Mat image_convert_bgr_to_nv21(const cv::Mat &image);

#endif //ESLAB_UTIL_IMAGE_HPP
