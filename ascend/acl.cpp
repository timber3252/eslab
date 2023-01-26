//
// Created by timber3252 on 2023/1/26.
//

#include "acl.hpp"

#include <stdexcept>

#include "acl/acl.h"
#include "atlasutil/atlas_model.h"
#include "atlasutil/atlas_utils.h"
#include "atlasutil/acl_device.h"

void aclrt_malloc(void **buffer, std::size_t size) {
  aclrtMalloc(buffer, size, ACL_MEM_MALLOC_HUGE_FIRST);

  if (buffer == nullptr)
    throw std::runtime_error("aclrt malloc failed");
}

void aclrt_free(void *buffer) {
  aclrtFree(buffer);
}
