//
// Created by timber3252 on 2023/1/28.
//

#ifndef ESLAB_FACE_RECOGNITION_FACE_UTILS_HPP
#define ESLAB_FACE_RECOGNITION_FACE_UTILS_HPP

#include <cmath>
#include <string>
#include <fstream>

#include <opencv2/opencv.hpp>

#include "face_detect.hpp"
#include "face_feature_mask.hpp"
#include "face_recognition.hpp"

class FaceUtils {
  static constexpr double kFaceMatchThreshold = 0.5;

public:
  struct Result {
    std::uint32_t index;
    cv::Mat face_image;
    std::string face_tag;
    double score;
  };

public:
  FaceUtils(const std::string &face_detection,
            const std::string &vanilla_cnn,
            const std::string &sphere_face,
            const std::string &face_library);

  std::map<std::uint32_t, Result> face_recognition(cv::Mat &frame);
  std::pair<bool, std::vector<double>> get_feature_vector(cv::Mat &frame);

private:
  std::map<std::string, std::vector<double>> face_library_{};
  FaceDetect face_detect_;
  FaceFeatureMask face_feature_mask_;
  FaceRecognition face_recognition_;
};

#endif //ESLAB_FACE_RECOGNITION_FACE_UTILS_HPP
