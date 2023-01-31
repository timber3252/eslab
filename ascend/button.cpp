//
// Created by timber3252 on 2023/1/31.
//

#include "button.hpp"

#include <thread>
#include <mutex>

extern bool flag;
extern std::mutex mtx;

std::uint8_t Button::button_read_s2() {
  std::uint8_t res;

  thirdparty::Keyhandler key;
  thirdparty::Key_Init(thirdparty::S3, &key);
  res = thirdparty::Key_Status(&key);
  thirdparty::Key_close(&key);

  return res;
}

Button::Button() {
  button_ = (thirdparty::Button*) malloc(sizeof(thirdparty::Button));
  memset(button_, 0x00, sizeof(thirdparty::Button));

  thirdparty::button_init(button_, button_read_s2, 0, 1);
  thirdparty::button_attach(button_, thirdparty::PRESS_DOWN, button_callback);
  thirdparty::button_start(button_);

  std::thread button_thread([&]() {
    while (true) {
      thirdparty::button_ticks();
      usleep(5000);
    }
  });

  button_thread.detach();
}

void Button::button_callback(void *args) {
  mtx.lock();
  flag = true;
  mtx.unlock();
}
