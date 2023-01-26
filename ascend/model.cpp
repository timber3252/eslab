//
// Created by timber3252 on 2023/1/26.
//

#include "model.hpp"

#include <stdexcept>
#include <vector>

Model::Model(const std::string &model_path) : model_(model_path) {}

void Model::init() {
  auto res = model_.Init();

  if (res)
    throw std::runtime_error("model init failed");
}

void Model::create_input(void *buffer, std::size_t size) {
  auto res = model_.CreateInput(buffer, size);

  if (res)
    throw std::runtime_error("create input buffer failed");
}

void Model::destroy_resource() {
  model_.DestroyResource();
}

std::vector<InferenceOutput> Model::execute() {
  std::vector<InferenceOutput> infer_outputs;
  auto ret = model_.Execute(infer_outputs);

  if (ret)
    throw std::runtime_error("execute model inference failed");

  return infer_outputs;
}
