//
// Created by timber3252 on 2023/1/31.
//

#ifndef ESLAB_ASCEND_BUTTON_HPP
#define ESLAB_ASCEND_BUTTON_HPP

#include <cstdint>
#include "3rdparty/key.hpp"
#include "3rdparty/multi_button.hpp"

class Button {
public:
  Button();

private:
  static std::uint8_t button_read_s2();
  static void button_callback(void *args);

  thirdparty::Button *button_;
};

#endif //ESLAB_ASCEND_BUTTON_HPP
