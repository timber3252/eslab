//
// Created by timber3252 on 2023/1/28.
//

#include "math.hpp"

double dot(const std::vector<double> &lhs, const std::vector<double> &rhs) {
  if (lhs.size() != rhs.size()) {
    throw std::runtime_error("vector size is not equal when calculating dot product");
  }

  return std::inner_product(lhs.begin(), lhs.end(), rhs.begin(), 0.0);
}

double norm(const std::vector<double> &vec) {
  return std::sqrt(dot(vec, vec));
}

double cosine(const std::vector<double> &lhs, const std::vector<double> &rhs) {
  return dot(lhs, rhs) / norm(lhs) / norm(rhs);
}

std::vector<double> to_double_vector(const std::vector<float> &vec) {
  std::vector<double> result;
  for (auto &i : vec) result.emplace_back(i);
  return result;
}

