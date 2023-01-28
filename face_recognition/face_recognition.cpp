//
// Created by timber3252 on 2023/1/27.
//

#include "face_recognition.hpp"

#include "../ascend/acl.hpp"
#include "../util/image.hpp"

FaceRecognition::FaceRecognition(const std::string &model_path)
: input_buffer_(nullptr), model_(model_path) {
  init();
}

FaceRecognition::~FaceRecognition() {
  aclrt_free(input_buffer_);
  model_.destroy_resource();
}

void FaceRecognition::init() {
  model_.init();
  aclrt_malloc(&input_buffer_, kModelInputSize);
  model_.create_input(input_buffer_, kModelInputSize);
}

std::vector<FaceRecognition::Result> FaceRecognition::inference(const std::vector<FaceFeatureMask::Result> &feature_mask_result) {
  if (feature_mask_result.empty()) {
    // just ignore this image
    return std::vector<Result>{};
  }

  auto align_results = face_align(feature_mask_result);
  if (align_results.empty()) {
    // just ignore this frame
    return std::vector<Result>{};
  }

  std::vector<Result> results;

  // prepare input data (tensor with size 8 * 112 * 96 * 3)
  std::size_t input_size = align_results.size();
  std::size_t last_batch_size = input_size % kModelBatch;
  std::size_t batch_count = input_size / kModelBatch + !!last_batch_size;

  for (std::size_t i = 0; i < batch_count; ++i) {
    std::size_t start_index = i * kModelBatch;
    auto pos = reinterpret_cast<std::uint8_t*>(input_buffer_);

    for (std::size_t j = 0; j < kModelBatch; ++j) {
      // for the last batch, fulfill the extra data with the last image as placeholder
      auto input = align_results[std::min(start_index + j, input_size - 1)];

      memcpy(pos, input.aligned_face.ptr<std::uint8_t>(), kModelImageSize);
      pos += kModelImageSize;
      memcpy(pos, input.aligned_flip_face.ptr<std::uint8_t>(), kModelImageSize);
      pos += kModelImageSize;
    }

    auto result = model_.execute();
    auto data = reinterpret_cast<float*>(result[0].data.get());

    for (std::size_t j = 0; j < kModelBatch; ++j) {
      std::size_t index = start_index + j;
      if (index >= input_size) break;
      auto input = align_results[index];

      results.emplace_back(Result{
        .index = input.index,
        .face = input.face,
        .feature_vector = {}
      });

      std::size_t st = j * kFeatureVectorLength;
      for (std::size_t k = 0; k < kFeatureVectorLength; ++k) {
        results.back().feature_vector.emplace_back(data[st + k]);
      }
    }
  }

  return results;
}

std::vector<FaceRecognition::AlignResult> FaceRecognition::face_align(const std::vector<FaceFeatureMask::Result> &src) {
  std::vector<AlignResult> results;

  auto check_transform = [](const cv::Mat &mat) {
    return (mat.type() == CV_32F || mat.type() == CV_64F) && mat.rows == 2 && mat.cols == 3;
  };

  std::for_each(src.begin(), src.end(), [&](const FaceFeatureMask::Result &data) {
    double x_scale = 1.0 * kModelWidth / data.face_image.cols;
    double y_scale = 1.0 * kModelHeight / data.face_image.rows;

    auto resize = image_resize(data.face_image, kModelWidth, kModelHeight);

    const std::vector<cv::Point2f> dst_points{
      cv::Point2f(kLeftEyeX, kLeftEyeY),
      cv::Point2f(kRightEyeX, kRightEyeY),
      cv::Point2f(kNoseX, kNoseY),
      cv::Point2f(kLeftMouthCornerX, kLeftMouthCornerY),
      cv::Point2f(kRightMouthCornerX, kRightMouthCornerY)
    };

    const std::vector<cv::Point2f> src_points{
      cv::Point2d(data.left_eye.x * x_scale, data.left_eye.y * y_scale),
      cv::Point2d(data.right_eye.x * x_scale, data.right_eye.y * y_scale),
      cv::Point2d(data.nose.x * x_scale, data.nose.y * y_scale),
      cv::Point2d(data.left_mouth.x * x_scale, data.right_mouth.y * y_scale),
      cv::Point2d(data.right_mouth.x * x_scale, data.right_mouth.y * y_scale)
    };

    // get transform matrix
    cv::Mat trans_matrix = cv::estimateAffinePartial2D(src_points, dst_points);
    if (!check_transform(trans_matrix)) {
      trans_matrix = cv::estimateAffine2D(src_points, dst_points);

      if (!check_transform(trans_matrix)) {
        // just ignore this image
        return;
      }
    }

    // apply transform
    cv::Mat aligned_image;
    cv::warpAffine(resize, aligned_image, trans_matrix, cv::Size(kModelWidth, kModelHeight));

    // flip
    cv::Mat aligned_flip_image = image_flip_vertically(aligned_image);

    // bgr2rgb for aipp
    aligned_image = image_convert_bgr_to_rgb(aligned_image);
    aligned_flip_image = image_convert_bgr_to_rgb(aligned_flip_image);

    results.emplace_back(AlignResult{
     .index = data.index,
     .face = data.face_image,
     .aligned_face = aligned_image,
     .aligned_flip_face = aligned_flip_image
    });
  });

  return results;
}
