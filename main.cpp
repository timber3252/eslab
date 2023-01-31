//
// Created by timber3252 on 2023/1/26.
//

#include <iostream>
#include <thread>
#include <unistd.h>

#include "ascend/acl_device.hpp"
#include "ascend/presenter.hpp"
#include "ascend/pca_9557.hpp"
#include "face_recognition/face_utils.hpp"

int main() {
  AclDeviceRAII acl_device;
  FaceUtils face_utils("../data/face_detection.om",
                       "../data/vanillacnn.om",
                       "../data/sphereface.om",
                       "../data/faces");
  Presenter presenter("../data/face_recognition.conf");
  Pca9557 pca9557;

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

    auto ret = face_utils.face_recognition(frame);

    // send labeled image to huawei presenter
    presenter.SendImage(ret.second);

    const auto &results = ret.first;

    std::int32_t count_unknown = static_cast<std::int32_t>(
      std::count_if(results.begin(), results.end(), [&](const auto &data) {
        return data.second.face_tag == "unknown";
      })
    );
    std::int32_t count_known = static_cast<std::int32_t>(results.size()) - count_unknown;

    // show unknown faces count in the left, known faces count in the right on PCA9557
    pca9557.show_number(count_unknown * 100 + count_known);
  }

  return 0;
}
