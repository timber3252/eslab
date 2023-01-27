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

cv::Mat image_crop(const cv::Mat &image, std::int32_t row_start, std::int32_t row_end,
                   std::int32_t col_start, std::int32_t col_end) {
  return image(cv::Range(row_start, row_end), cv::Range(col_start, col_end));
}
