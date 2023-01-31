//
// Created by timber3252 on 2023/1/31.
//

#ifndef ESLAB_ASCEND_SSD_1306_HPP
#define ESLAB_ASCEND_SSD_1306_HPP

#include <opencv2/opencv.hpp>
#include <thread>

#include "../face_recognition/face_utils.hpp"

class Ssd1306 {
  static constexpr std::uint32_t kWidth = 128;
  static constexpr std::uint32_t kHeight = 64;

public:
  Ssd1306();
  void show_string(std::uint32_t line, const std::string &s);
  void refresh();
};

#endif //ESLAB_ASCEND_SSD_1306_HPP
