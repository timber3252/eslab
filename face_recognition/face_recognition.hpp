//
// Created by timber3252 on 2023/1/27.
//

#ifndef ESLAB_FACE_RECOGNITION_FACE_RECOGNITION_HPP
#define ESLAB_FACE_RECOGNITION_FACE_RECOGNITION_HPP

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "../ascend/model.hpp"
#include "face_feature_mask.hpp"

class FaceRecognition {
  static constexpr std::uint32_t kModelBatch = 4;
  static constexpr std::uint32_t kEachBatchImageCount = 2;
  static constexpr std::uint32_t kModelWidth = 96;
  static constexpr std::uint32_t kModelHeight = 112;
  static constexpr std::uint32_t kModelImageSize = kModelWidth * kModelHeight * 3;
  static constexpr std::uint32_t kModelInputSize = kModelBatch * kEachBatchImageCount * kModelImageSize;

  static constexpr float kLeftEyeX = 30.2946;
  static constexpr float kLeftEyeY = 51.6963;
  static constexpr float kRightEyeX = 65.5318;
  static constexpr float kRightEyeY = 51.5014;
  static constexpr float kNoseX = 48.0252;
  static constexpr float kNoseY = 71.7366;
  static constexpr float kLeftMouthCornerX = 33.5493;
  static constexpr float kLeftMouthCornerY = 92.3655;
  static constexpr float kRightMouthCornerX = 62.7299;
  static constexpr float kRightMouthCornerY = 92.2041;

public:
  static constexpr std::uint32_t kFeatureVectorLength = 1024;

  struct Result {
    std::uint32_t index;
    cv::Mat face;
    std::vector<float> feature_vector;
  };

private:
  struct AlignResult {
    std::uint32_t index;
    cv::Mat face, aligned_face, aligned_flip_face;
  };

public:
  explicit FaceRecognition(const std::string &model_path);
  ~FaceRecognition();

  std::vector<Result> inference(const std::vector<FaceFeatureMask::Result> &feature_mask_result);

private:
  void init();
  std::vector<AlignResult> face_align(const std::vector<FaceFeatureMask::Result> &feature_mask_result);

  void *input_buffer_;
  Model model_;
};

#endif //ESLAB_FACE_RECOGNITION_FACE_RECOGNITION_HPP
