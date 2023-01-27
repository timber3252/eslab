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
  model_.destroy_resource();
}

void FaceRecognition::init() {
  model_.init();
  aclrt_malloc(&input_buffer_, kModelInputSize);
  model_.create_input(input_buffer_, kModelInputSize);
}

std::vector<FaceRecognition::Result> FaceRecognition::inference(const std::vector<FaceFeatureMask::Result> &feature_mask_result) {
  if (feature_mask_result.empty()) {
    throw std::runtime_error("input inference data is empty");
  }

  auto align_results = face_align(feature_mask_result);
  if (align_results.empty()) {
    // TODO: maybe ignore error
    throw std::runtime_error("align results is empty");
  }

  // TODO: prepare input

  auto result = model_.execute();

  // TODO: post process
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
        // TODO: maybe ignore error
        throw std::runtime_error("determination of transform matrix failed");
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
     .aligned_face = aligned_image,
     .aligned_flip_face = aligned_flip_image
    });
  });

  return results;
}
