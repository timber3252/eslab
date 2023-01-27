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
  cv::Mat mean(kModelHeight, kModelWidth, CV_8UC3, (void*)(kTrainMean));
  cv::Mat std(kModelHeight, kModelWidth, CV_8UC3, (void*)(kTrainStd));

  mean.convertTo(train_mean_, CV_32FC3, 1.0 / 255.0);
  std.convertTo(train_std_, CV_32FC3, 1.0 / 255.0);

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
    cv::Mat resize = image_resize(crop, kModelWidth, kModelHeight);
    cv::Mat input = image_convert_8uc3_to_32fc3(resize);

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
      auto input_32f = inputs[std::min(start_index + j, input_size - 1)];
      auto split_input = image_split_channel(input_32f);

      for (auto &k : split_input) {
        memcpy(pos, k.ptr<float>(), kModelImageScale * sizeof(float));
        pos += kModelImageScale;
      }
    }

    // do inference
    auto result = model_.execute();
    auto data = reinterpret_cast<float*>(result[0].data.get());

    // process output data
    for (std::size_t j = 0; j < kModelBatch; ++j) {
      std::size_t index = start_index + j;
      if (index >= input_size) break;
      std::size_t st = j * 10;

      auto get_coords = [&](float x, float y) -> cv::Point {
        return {
          static_cast<std::int32_t>(1.0 * (x + kNormalizedCenterData) * results[index].face_image.cols),
          static_cast<std::int32_t>(1.0 * (y + kNormalizedCenterData) * results[index].face_image.rows)
        };
      };

      results[index].left_eye = get_coords(data[st + LEFT_EYE_X], data[st + LEFT_EYE_Y]);
      results[index].right_eye = get_coords(data[st + RIGHT_EYE_X], data[st + RIGHT_EYE_Y]);
      results[index].nose = get_coords(data[st + NOSE_X], data[st + NOSE_Y]);
      results[index].left_mouth = get_coords(data[st + LEFT_MOUTH_X], data[st + LEFT_MOUTH_Y]);
      results[index].right_mouth = get_coords(data[st + RIGHT_MOUTH_X], data[st + RIGHT_MOUTH_Y]);

      cv::circle(results[index].face_image, results[index].left_eye, 1, cv::Scalar(255, 0, 0));
      cv::circle(results[index].face_image, results[index].right_eye, 1, cv::Scalar(255, 0, 0));
      cv::circle(results[index].face_image, results[index].nose, 1, cv::Scalar(0, 255, 0));
      cv::circle(results[index].face_image, results[index].left_mouth, 1, cv::Scalar(0, 0, 255));
      cv::circle(results[index].face_image, results[index].right_mouth, 1, cv::Scalar(0, 0, 255));

      cv::imwrite("output.jpg", results[index].face_image);
    }
  }

  // debug
  exit(-1);

  // TODO: Post Process
}
