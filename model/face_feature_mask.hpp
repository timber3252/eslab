//
// Created by timber3252 on 2023/1/26.
//

#ifndef ESLAB_MODEL_FACE_FEATURE_MASK_HPP
#define ESLAB_MODEL_FACE_FEATURE_MASK_HPP

#include <string>
#include <opencv2/opencv.hpp>

#include "../ascend/model.hpp"
#include "face_detect.hpp"

class FaceFeatureMask {
  static constexpr std::uint32_t kModelBatch = 4;
  static constexpr std::uint32_t kModelWidth = 40;
  static constexpr std::uint32_t kModelHeight = 40;
  static constexpr std::uint32_t kModelImageScale = kModelWidth * kModelHeight * 3;
  static constexpr std::uint32_t kModelInputSize = kModelBatch * kModelImageScale * sizeof(float);

  static constexpr float kNormalizedCenterData = 0.5;

public:
  struct Result {
    std::uint32_t index;
    cv::Mat face_image;
    cv::Point left_eye, right_eye, nose, left_mouth, right_mouth;
  };

private:
  enum FaceFeaturePos {
    kLeftEyeX = 0,
    kLeftEyeY = 1,
    kRightEyeX = 2,
    kRightEyeY = 3,
    kNoseX = 4,
    kNoseY = 5,
    kLeftMouthX = 6,
    kLeftMouthY = 7,
    kRightMouthX = 8,
    kRightMouthY = 9
  };

public:
  explicit FaceFeatureMask(const std::string &model_path);
  ~FaceFeatureMask();

  std::vector<Result> inference(const cv::Mat &frame, const std::vector<FaceDetect::Result> &detect_result);

private:
  void init();

  void *input_buffer_;
  Model model_;

  cv::Mat train_mean_{}, train_std_{};
};

#endif //ESLAB_MODEL_FACE_FEATURE_MASK_HPP
