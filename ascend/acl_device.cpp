//
// Created by timber3252 on 2023/1/26.
//

#include "acl_device.hpp"

#include <stdexcept>

AclDeviceRAII::AclDeviceRAII() {
  auto ret = acl_dev_.Init();
  if (ret) {
    throw std::runtime_error("init acl resource failed");
  }
}
