//
// Created by timber3252 on 2023/1/31.
//

#include "ssd_1306.hpp"

#include "../util/image.hpp"
#include "3rdparty/ssd1306.hpp"

Ssd1306::Ssd1306() {
  thirdparty::oled_init();

  for (std::size_t i = 0; i <= 4; ++i) {
    show_string(i, std::string(21, ' '));
  }
}

void Ssd1306::refresh() {
  thirdparty::Refresh();
}

void Ssd1306::show_string(std::uint32_t line, const std::string &s) {
  thirdparty::ShowString(0, line * 12, s.c_str(), thirdparty::size1206);
}

