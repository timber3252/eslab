//
// Created by timber3252 on 2023/1/26.
//

#ifndef ESLAB_ASCEND_ACL_HPP
#define ESLAB_ASCEND_ACL_HPP

#include <cstdint>

void aclrt_malloc(void **buffer, std::size_t size);
void aclrt_free(void *buffer);

#endif //ESLAB_ASCEND_ACL_HPP
