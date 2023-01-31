//
// Created by timber3252 on 2023/1/31.
//

#ifndef ESLAB_ASCEND_SSD_1306_HPP
#define ESLAB_ASCEND_SSD_1306_HPP

#include <opencv2/opencv.hpp>

class Ssd1306 {
public:
  Ssd1306();
  void show_image(const cv::Mat &image);
};

#endif //ESLAB_ASCEND_SSD_1306_HPP
