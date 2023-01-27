//
// Created by timber3252 on 2023/1/26.
//

#include <iostream>

#include "ascend/acl_device.hpp"
#include "face_recognition/face_detect.hpp"
#include "face_recognition/face_feature_mask.hpp"
#include "face_recognition/face_recognition.hpp"

int main() {
  AclDeviceRAII acl_device;
  FaceDetect face_detect("../data/face_detection.om");
  FaceFeatureMask face_feature_mask("../data/vanillacnn.om");
  FaceRecognition face_recognition("../data/face_detection.om");

  cv::VideoCapture capture(0);
  if (!capture.isOpened()) {
    std::cerr << "video capture open error" << std::endl;
    return -1;
  }

  while (true) {
    cv::Mat frame;
    if (!capture.read(frame)) {
      std::cerr << "video capture failed" << std::endl;
      return -1;
    }

    auto detect_result = face_detect.inference(frame);
    auto feature_mask_result = face_feature_mask.inference(frame, detect_result);
    face_recognition.inference(feature_mask_result);
  }

  return 0;
}
