//
// Created by timber3252 on 2023/1/28.
//

#include "face_utils.hpp"

#include "../util/math.hpp"

FaceUtils::FaceUtils(const std::string &face_detection,
                     const std::string &vanilla_cnn,
                     const std::string &sphere_face,
                     const std::string &face_library)
: face_detect_(face_detection), face_feature_mask_(vanilla_cnn), face_recognition_(sphere_face) {
}

std::map<std::uint32_t, FaceUtils::Result> FaceUtils::face_recognition(cv::Mat &frame) {
  std::map<std::uint32_t, Result> results{};

  auto detect_result = face_detect_.inference(frame);
  auto feature_mask_result = face_feature_mask_.inference(frame, detect_result);
  for (auto &fmr : feature_mask_result) {
    results[fmr.index] = {
        .index = fmr.index,
        .face_image = fmr.face_image,
        .face_tag = "unlabeled",
        .score = 0,
    };
  }

  auto face_recognition_result = face_recognition_.inference(feature_mask_result);

  for (auto &frr : face_recognition_result) {
    std::string highest_score_name = "unknown";
    double highest_score = 0;

    for (auto &face : face_library_) {
      // get the similarity of face in library and captured face
      double score = cosine(to_double_vector(frr.feature_vector), face.second);

      if (score < kFaceMatchThreshold)
        continue;

      if (score > highest_score) {
        highest_score = score;
        highest_score_name = face.first;
      }
    }

    results[frr.index].face_tag = highest_score_name;
    results[frr.index].score = highest_score;
  }

  return results;
}

std::pair<bool, std::vector<double>> FaceUtils::get_feature_vector(cv::Mat &frame) {
  auto detect_result = face_detect_.inference(frame);
  auto feature_mask_result = face_feature_mask_.inference(frame, detect_result);
  auto face_recognition_result = face_recognition_.inference(feature_mask_result);

  if (face_recognition_result.empty())
    return {false, {}};

  return {true, to_double_vector(face_recognition_result[0].feature_vector)};
}
