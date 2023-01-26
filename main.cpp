//
// Created by timber3252 on 2023/1/26.
//

#include <iostream>

#include "ascend/acl_device.hpp"
#include "model/face_detect.hpp"

int main() {
  AclDeviceRAII acl_device;

  FaceDetect face_detect("../data/face_detection.om");
  face_detect.init();

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

    face_detect.inference(frame);
  }

  return 0;
}
