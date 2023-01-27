//
// Created by timber3252 on 2023/1/26.
//

#include "image.hpp"

#include <stdexcept>

cv::Mat image_resize(const cv::Mat &image, std::int32_t width, std::int32_t height) {
  cv::Mat result;
  cv::resize(image, result, cv::Size(width, height));

  if (result.empty())
    throw std::runtime_error("image resize failed");

  return result;
}

// https://blog.csdn.net/u012815193/article/details/121818288
cv::Mat image_convert_bgr_to_nv21(const cv::Mat &image) {
  cv::Mat result(image.rows + image.rows / 2, image.cols, CV_8UC1);
  cv::cvtColor(image, result, cv::COLOR_BGR2YUV_I420);

  std::int32_t offset = image.rows * image.cols;
  std::int32_t len = offset / 2, half_len = len / 2;

  auto tmp = new std::uint8_t[len];
  auto data = result.data;

  memcpy(tmp, data + offset, len);
  for (int i = 0; i < half_len; ++i) {
    data[offset + i * 2] = tmp[half_len + i];
    data[offset + i * 2 + 1] = tmp[i];
  }

  delete []tmp;
  return result;
}

cv::Mat image_convert_8uc3_to_32fc3(const cv::Mat &image) {
  cv::Mat result(image.rows, image.cols, CV_32FC3);
  image.convertTo(result, CV_32FC3, 1.0 / 255.0);
  return result;
}

cv::Mat image_crop(const cv::Mat &image, std::int32_t row_start, std::int32_t row_end,
                   std::int32_t col_start, std::int32_t col_end) {
  row_start = (row_start >> 1) << 1;
  col_start = (col_start >> 1) << 1;

  row_end = ((row_end >> 1) << 1) - 1;
  col_end = ((col_end >> 1) << 1) - 1;

  return image(cv::Range(row_start, row_end), cv::Range(col_start, col_end));
}

std::vector<cv::Mat> image_split_channel(const cv::Mat &image) {
  std::vector<cv::Mat> split_input;
  cv::split(image, split_input);
  return split_input;
}

