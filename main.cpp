//
// Created by timber3252 on 2023/1/26.
//

#include <iostream>
#include <unistd.h>

#include "ascend/acl_device.hpp"
#include "face_recognition/face_utils.hpp"

int main() {
  AclDeviceRAII acl_device;

  FaceUtils face_utils("../data/face_detection.om",
                       "../data/vanillacnn.om",
                       "../data/sphereface.om",
                       "../data/faces");

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

    auto result = face_utils.face_recognition(frame);
    if (!result.empty()) {
      std::cout << result.begin()->second.face_tag << " " << result.begin()->second.score << std::endl;
    } else {
      std::cout << "empty" << std::endl;
    }

    usleep(1000 * 1000);
  }

  return 0;
}
