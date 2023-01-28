//
// Created by timber3252 on 2023/1/26.
//

#include "face_detect.hpp"

#include <utility>

#include "../util/image.hpp"
#include "../ascend/acl.hpp"

FaceDetect::FaceDetect(const std::string& model_path)
: input_buffer_(nullptr), model_(model_path) {
  init();
}

FaceDetect::~FaceDetect() {
  aclrt_free(input_buffer_);
  model_.destroy_resource();
}

void FaceDetect::init() {
  model_.init();
  aclrt_malloc(&input_buffer_, kModelInputSize);
  model_.create_input(input_buffer_, kModelInputSize);
}

std::vector<FaceDetect::Result> FaceDetect::inference(const cv::Mat &frame) {
  // prepare model input
  cv::Mat input = image_convert_bgr_to_nv21(image_resize(frame, kModelWidth, kModelHeight));
  memcpy(input_buffer_, input.ptr<void>(), kModelInputSize);

  // do inference
  auto result = model_.execute();

  // produce output
  auto total = reinterpret_cast<std::uint32_t*>(result[0].data.get())[0];
  auto data = reinterpret_cast<float*>(result[1].data.get());

  std::vector<Result> ret;

  for (auto i = 0; i < total; ++i) {
    Result res;

    auto score = static_cast<std::uint32_t>(data[SCORE + i * kItemSize] * 100);
    if (score < 70) break;

    cv::Point left_top, right_bottom;
    left_top.x = data[TOP_LEFT_X + i * kItemSize] * frame.cols;
    left_top.y = data[TOP_LEFT_Y + i * kItemSize] * frame.rows;
    right_bottom.x = data[BOTTOM_RIGHT_X + i * kItemSize] * frame.cols;
    right_bottom.y = data[BOTTOM_RIGHT_Y + i * kItemSize] * frame.rows;

    auto index = data[LABEL + i * kItemSize];

    res.index = i;
    res.score = score;
    res.left_top = left_top;
    res.right_bottom = right_bottom;

    ret.emplace_back(res);
  }

  return ret;
}
