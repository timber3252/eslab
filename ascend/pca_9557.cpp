//
// Created by timber3252 on 2023/1/31.
//

#include "pca_9557.hpp"
#include "3rdparty/pca9557.hpp"
#include <thread>

Pca9557::Pca9557() {
  thirdparty::pca9557_init("/dev/i2c-1");
  thirdparty::pca9557_setnum(0, 0, 0, 0);
  thread_pca9557_ = std::thread(thirdparty::pca9557_show);
  thread_pca9557_.detach();
}

void Pca9557::show_number(int x) {
  int a = (x / 1000) % 10;
  int b = (x / 100) % 10;
  int c = (x / 10) % 10;
  int d = x % 10;

  thirdparty::pca9557_setnum(a, b, c, d);
}
