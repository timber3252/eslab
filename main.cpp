//
// Created by timber3252 on 2023/1/26.
//

#include <iostream>
#include <thread>
#include <unistd.h>

#include "ascend/acl_device.hpp"
#include "ascend/presenter.hpp"
#include "ascend/pca_9557.hpp"
#include "ascend/ssd_1306.hpp"
#include "ascend/button.hpp"
#include "face_recognition/face_utils.hpp"

// button interrupt flag
bool flag = false;
std::mutex mtx;

int main() {
  AclDeviceRAII acl_device;
  FaceUtils face_utils("../data/face_detection.om",
                       "../data/vanillacnn.om",
                       "../data/sphereface.om",
                       "../data/faces");
  Presenter presenter("../data/face_recognition.conf");
  Pca9557 pca9557;
  Ssd1306 ssd1306;
  Button button;

  // opencv camera
  cv::VideoCapture capture(0);
  if (!capture.isOpened()) {
    std::cerr << "video capture open error" << std::endl;
    return -1;
  }

  while (true) {
    // capture frames here to make sure the lastest images are got when interruption
    cv::Mat frame;
    if (!capture.read(frame)) {
      std::cerr << "video capture failed" << std::endl;
      return -1;
    }

    mtx.lock();
    if (flag) {
      auto ret = face_utils.face_recognition(frame);

      // send labeled image to huawei presenter
      presenter.SendImage(ret.second);

      const auto &results = ret.first;

      std::int32_t count_unknown = 0, count_known = 0;
      std::vector<std::string> found_list;

      for (auto &i : results) {
        if (i.second.face_tag == "unknown") {
          ++count_unknown;
        } else {
          ++count_known;
          found_list.emplace_back(i.second.face_tag);
        }
      }

      // show result on ssd1306
      if (count_known > 0) {
        ssd1306.show_string(0, "valid: true ");
        for (std::size_t i = 1; i <= 4; ++i) {
          ssd1306.show_string(i, std::string(21, ' '));
          if (i <= found_list.size())
            ssd1306.show_string(i, "found: " + found_list[i - 1]);
        }
      } else {
        ssd1306.show_string(0, "valid: false");
        for (std::size_t i = 1; i <= 4; ++i) {
          ssd1306.show_string(i, std::string(21, ' '));
        }
      }

      // show unknown faces count in the left, known faces count in the right on PCA9557
      pca9557.show_number(count_unknown * 100 + count_known);
      ssd1306.refresh();

      flag = false;
    }
    mtx.unlock();

    usleep(10000);
  }

  return 0;
}
