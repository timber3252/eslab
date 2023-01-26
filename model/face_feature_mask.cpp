//
// Created by timber3252 on 2023/1/26.
//

#include "face_feature_mask.hpp"

#include "../ascend/acl.hpp"
#include "../util/image.hpp"
#include "../ascend/face_feature_train_mean.hpp"
#include "../ascend/face_feature_train_std.hpp"

FaceFeatureMask::FaceFeatureMask(const std::string &model_path)
: input_buffer_(nullptr), model_(model_path),
  train_mean_(kModelHeight, kModelWidth, CV_32FC3, (void *)kTrainMean),
  train_std_(kModelHeight, kModelWidth, CV_32FC3, (void *)kTrainStd) {
  if (train_mean_.empty() || train_std_.empty()) {
    throw std::runtime_error("load mean / std matrix failed");
  }
}

FaceFeatureMask::~FaceFeatureMask() {
  model_.destroy_resource();
}

void FaceFeatureMask::init() {
  model_.init();
  aclrt_malloc(&input_buffer_, kModelInputSize);
  model_.create_input(input_buffer_, kModelInputSize);
}

std::vector<FaceFeatureMask::Result> FaceFeatureMask::inference(cv::Mat &frame) {
  // TODO: Batch 4

  // prepare model input
  cv::Mat input = image_resize(frame, kModelWidth, kModelHeight);

  // normalization
  input = input - train_mean_;
  input = input / train_std_;
  if (input.empty())
    throw std::runtime_error("all the data is empty");

  // copy to buffer
  memcpy(input_buffer_, input.ptr<void>(), kModelInputSize);

  // do inference
  auto result = model_.execute();

  // TODO: Post Process
}
