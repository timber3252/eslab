//
// Created by timber3252 on 2023/1/26.
//

#ifndef ESLAB_MODEL_FACE_DETECT_HPP
#define ESLAB_MODEL_FACE_DETECT_HPP

#include <string>
#include <opencv2/opencv.hpp>

#include "../ascend/model.hpp"

class FaceDetect {
  static constexpr std::uint32_t kItemSize = 8;
  static constexpr std::uint32_t kModelWidth = 304;
  static constexpr std::uint32_t kModelHeight = 300;
  static constexpr std::uint32_t kModelInputSize = kModelWidth * kModelHeight * 3 / 2;

public:
  struct Result {
    std::uint32_t index, score;
    cv::Point left_top, right_bottom;
  };

private:
  enum BoxIndex {
    EMPTY = 0,
    LABEL = 1,
    SCORE = 2,
    TOP_LEFT_X = 3,
    TOP_LEFT_Y = 4,
    BOTTOM_RIGHT_X = 5,
    BOTTOM_RIGHT_Y = 6
  };

public:
  explicit FaceDetect(const std::string& model_path);
  ~FaceDetect();

  std::vector<Result> inference(const cv::Mat &frame);

private:
  void init();

  void *input_buffer_;
  Model model_;
};

#endif //ESLAB_MODEL_FACE_DETECT_HPP
