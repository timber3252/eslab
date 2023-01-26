//
// Created by timber3252 on 2023/1/26.
//

#ifndef ESLAB_ASCEND_MODEL_HPP
#define ESLAB_ASCEND_MODEL_HPP

#include <cstdint>
#include <string>

#include "acl/acl.h"
#include "atlasutil/atlas_utils.h"
#include "atlasutil/atlas_model.h"

class Model {
public:
  explicit Model(const std::string &model_path);
  void init();
  void create_input(void *buffer, std::size_t size);
  std::vector<InferenceOutput> execute();
  void destroy_resource();

private:
  AtlasModel model_;
};

#endif //ESLAB_ASCEND_MODEL_HPP
