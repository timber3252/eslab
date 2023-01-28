//
// Created by timber3252 on 2023/1/28.
//

#include "presenter.hpp"

#include <opencv2/opencv.hpp>

using namespace ascend::presenter;

Presenter::Presenter(const std::string &conf) : channel_(nullptr) {
  auto ret = OpenChannelByConfig(channel_, conf.data());
  if (ret != PresenterErrorCode::kNone) {
    throw std::runtime_error("open channel failed");
  }
}

void Presenter::SendImage(cv::Mat &image) {
  std::vector<std::uint8_t> encode_image;
  cv::imencode(".jpg", image, encode_image, {cv::IMWRITE_JPEG_QUALITY, 95});

  ImageFrame image_param;
  image_param.format = ImageFormat::kJpeg;
  image_param.width = image.cols;
  image_param.height = image.rows;
  image_param.size = encode_image.size();
  image_param.data = reinterpret_cast<std::uint8_t*>(encode_image.data());

  PresenterErrorCode error_code = PresentImage(channel_, image_param);
  if (error_code != PresenterErrorCode::kNone) {
    throw std::runtime_error("presenter image failed");
  }
}
