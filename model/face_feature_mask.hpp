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
  static constexpr std::uint32_t kModelImageSize = kModelWidth * kModelHeight * 3;
  static constexpr std::uint32_t kModelInputSize = kModelBatch * kModelImageSize;

public:
  struct Result {
    std::uint32_t index;
  };

public:
  explicit FaceFeatureMask(const std::string &model_path);
  ~FaceFeatureMask();

  void init();
  std::vector<Result> inference(const cv::Mat &frame, const std::vector<FaceDetect::Result> &detect_result);

private:
  void *input_buffer_;
  Model model_;

  cv::Mat train_mean_, train_std_;
};

#endif //ESLAB_MODEL_FACE_FEATURE_MASK_HPP
