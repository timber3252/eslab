//
// Created by timber3252 on 2023/1/31.
//

#include "ssd_1306.hpp"

#include "../util/image.hpp"
#include "3rdparty/ssd1306.hpp"

Ssd1306::Ssd1306() {
  thirdparty::oled_init();
}

void Ssd1306::refresh() {
  thirdparty::Refresh();
}

void Ssd1306::show_string(std::uint32_t line, const std::string &s) {
  thirdparty::oled_clear();
  thirdparty::ShowString(0, line * 12, s.c_str(), thirdparty::size1206);
}

