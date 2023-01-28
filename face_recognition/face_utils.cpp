//
// Created by timber3252 on 2023/1/28.
//

#include "face_utils.hpp"

#include <dirent.h>
#include "../util/math.hpp"

FaceUtils::FaceUtils(const std::string &face_detection,
                     const std::string &vanilla_cnn,
                     const std::string &sphere_face,
                     const std::string &face_library)
: face_detect_(face_detection), face_feature_mask_(vanilla_cnn), face_recognition_(sphere_face) {
  load_faces_from_folder(face_library);
}

std::pair<std::map<std::uint32_t, FaceUtils::Result>, cv::Mat> FaceUtils::face_recognition(cv::Mat &frame) {
  std::map<std::uint32_t, Result> results{};

  cv::Mat overlay_image = frame;

  std::vector<FaceDetect::Result> detect_result = face_detect_.inference(frame);
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

  for (auto &dr : detect_result) {
    auto pt1 = dr.left_top, pt2 = dr.right_bottom;

    cv::rectangle(overlay_image, pt1, pt2, cv::Scalar(0, 255, 0));

    const auto &result = results[dr.index];
    cv::putText(overlay_image, result.face_tag, cv::Point(pt1.x + 5, pt1.y - 5),
                cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 2);
    cv::putText(overlay_image, std::to_string(result.score) + "%", cv::Point(pt1.x + 5, pt2.y - 5),
                cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 2);
  }

  return {results, overlay_image};
}

std::pair<bool, std::vector<double>> FaceUtils::get_feature_vector(cv::Mat &frame) {
  auto detect_result = face_detect_.inference(frame);
  auto feature_mask_result = face_feature_mask_.inference(frame, detect_result);
  auto face_recognition_result = face_recognition_.inference(feature_mask_result);

  if (face_recognition_result.empty())
    return {false, {}};

  return {true, to_double_vector(face_recognition_result[0].feature_vector)};
}

void FaceUtils::load_faces_from_folder(const std::string &face_library) {
  DIR *dir = opendir(face_library.data());
  if (dir == nullptr) {
    throw std::runtime_error("face library path given is not a valid directory");
  }

  std::string folder_path = face_library + (face_library.back() == '/' ? "" : "/");
  folder_path_ = folder_path;

  dirent *dp;
  while ((dp = readdir(dir)) != nullptr) {
    std::string file_name(dp->d_name);

    if (file_name == ".." || file_name == ".") {
      continue;
    }

    std::string file_path = folder_path + file_name;

    std::ifstream fin(file_path);
    if (!fin.is_open()) {
      continue;
    }

    std::cout << "loaded face " << file_name << std::endl;

    std::vector<double> feature_vector(FaceRecognition::kFeatureVectorLength);
    for (auto &x : feature_vector) fin >> x;

    face_library_.insert({file_name, feature_vector});
    fin.close();
  }

  closedir(dir);
}

bool FaceUtils::add_face(const std::string &name, const std::vector<double> &feature_vector) {
  // check if face name already exist
  if (face_library_.count(name))
    return false;

  // update
  face_library_[name] = feature_vector;

  // write to file
  std::ofstream fout(folder_path_ + name);
  for (auto &x : feature_vector) {
    fout << std::setprecision(10) << std::fixed << x << " ";
  }

  std::cout << "added face: " << name << std::endl;

  fout.close();
}
