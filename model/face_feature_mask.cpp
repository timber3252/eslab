//
// Created by timber3252 on 2023/1/26.
//

#include "face_feature_mask.hpp"

#include "../ascend/acl.hpp"
#include "../util/image.hpp"
#include "../ascend/face_feature_train_mean.hpp"
#include "../ascend/face_feature_train_std.hpp"

FaceFeatureMask::FaceFeatureMask(const std::string &model_path)
: input_buffer_(nullptr), model_(model_path) {
  cv::Mat mean(kModelHeight, kModelWidth, CV_32FC3, (void *)kTrainMean);
  cv::Mat std(kModelHeight, kModelWidth, CV_32FC3, (void *)kTrainStd);

  // implicit convert 32fc3 to 8uc3
  train_mean_ = mean, train_std_ = std;

  if (train_mean_.empty() || train_std_.empty()) {
    throw std::runtime_error("load mean / std matrix failed");
  }

  init();
}

FaceFeatureMask::~FaceFeatureMask() {
  model_.destroy_resource();
}

void FaceFeatureMask::init() {
  model_.init();
  aclrt_malloc(&input_buffer_, kModelInputSize);
  model_.create_input(input_buffer_, kModelInputSize);
}

std::vector<FaceFeatureMask::Result> FaceFeatureMask::inference(const cv::Mat &frame,
                                                                const std::vector<FaceDetect::Result> &detect_result) {
  if (detect_result.empty()) {
    throw std::runtime_error("input inference data is empty");
  }

  std::vector<cv::Mat> inputs;
  std::vector<FaceFeatureMask::Result> results;

  std::for_each(detect_result.begin(), detect_result.end(), [&](const FaceDetect::Result &data) {
    // image preprocess
    cv::Mat crop = image_crop(frame,
                              data.left_top.y, data.right_bottom.y,
                              data.left_top.x, data.right_bottom.x);

    cv::Mat input = image_convert_8uc3_to_32fc3(image_resize(crop, kModelWidth, kModelHeight));

    // normalization
    input = input - train_mean_;
    input = input / train_std_;

    if (input.empty()) {
      throw std::runtime_error("all the data is empty");
    }

    inputs.emplace_back(input);
    results.emplace_back(Result{
      .index = data.index,
      .face_image = crop,
    });
  });

  // prepare input data (tensor with size 4 * 40 * 40 * 3)
  std::size_t input_size = inputs.size();
  std::size_t last_batch_size = input_size % kModelBatch;
  std::size_t batch_count = input_size / kModelBatch + !!last_batch_size;

  for (std::size_t i = 0; i < batch_count; ++i) {
    std::size_t start_index = i * kModelBatch;
    auto pos = reinterpret_cast<float*>(input_buffer_);

    for (std::size_t j = 0; j < kModelBatch; ++j) {
      // for the last batch, fulfill the extra data with the last image as placeholder
      auto input_32f = inputs[std::min(start_index + j, inputs.size() - 1)];

      memcpy(pos + j * kModelImageScale, input_32f.ptr<float>(), kModelImageScale * sizeof(float));
    }

    // do inference
    auto result = model_.execute();

    auto total = reinterpret_cast<std::uint32_t*>(result[1].data.get())[0];
    auto data = reinterpret_cast<float*>(result[0].data.get());

    // debug
    std::cout << total << std::endl;
  }

  // debug
  exit(-1);

  // TODO: Post Process
}
