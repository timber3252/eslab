//
// Created by timber3252 on 2023/1/26.
//

#include <iostream>

#include "model/face_detect.hpp"
#include "atlasutil/acl_device.h"

int main() {
  AclDevice aclDev;
  auto ret = aclDev.Init();
  if (ret) {
    std::cerr << "init resource failed" << std::endl;
    return -1;
  }

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
