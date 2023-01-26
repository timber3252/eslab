//
// Created by timber3252 on 2023/1/26.
//

#ifndef ESLAB_MODEL_FACE_FEATURE_MASK_HPP
#define ESLAB_MODEL_FACE_FEATURE_MASK_HPP

#include <string>
#include <opencv2/opencv.hpp>

#include "../ascend/model.hpp"

class FaceFeatureMask {
  static constexpr std::uint32_t kModelBatch = 4;
  static constexpr std::uint32_t kModelWidth = 40;
  static constexpr std::uint32_t kModelHeight = 40;
  static constexpr std::uint32_t kModelInputSize = kModelWidth * kModelHeight * 3;

public:
  struct Result {

  };

public:
  explicit FaceFeatureMask(const std::string &model_path);
  ~FaceFeatureMask();

  void init();
  std::vector<Result> inference(cv::Mat &frame);

private:
  void *input_buffer_;
  Model model_;

  cv::Mat train_mean_, train_std_;
};

#endif //ESLAB_MODEL_FACE_FEATURE_MASK_HPP
