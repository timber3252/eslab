//
// Created by timber3252 on 2023/1/31.
//

#include "ssd_1306.hpp"

#include "../util/image.hpp"
#include "3rdparty/ssd1306.hpp"

Ssd1306::Ssd1306() {
  thirdparty::oled_init();
}

void Ssd1306::show_image(const cv::Mat &image) {
  cv::Mat resize = image_resize(image, 128, 64);

//  thirdparty::oled_showPicture();
}

