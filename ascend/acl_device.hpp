//
// Created by timber3252 on 2023/1/26.
//

#ifndef ESLAB_ASCEND_ACL_DEVICE_HPP
#define ESLAB_ASCEND_ACL_DEVICE_HPP

#include "atlasutil/acl_device.h"

class AclDeviceRAII {
public:
  AclDeviceRAII();

private:
  AclDevice acl_dev_{};
};

#endif //ESLAB_ASCEND_ACL_DEVICE_HPP
