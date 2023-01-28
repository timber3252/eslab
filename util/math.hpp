//
// Created by timber3252 on 2023/1/28.
//

#ifndef ESLAB_UTIL_MATH_HPP
#define ESLAB_UTIL_MATH_HPP

#include <cmath>
#include <vector>
#include <stdexcept>
#include <numeric>

double dot(const std::vector<double> &lhs, const std::vector<double> &rhs);
double norm(const std::vector<double> &vec);
double cosine(const std::vector<double> &lhs, const std::vector<double> &rhs);
std::vector<double> to_double_vector(const std::vector<float> &vec);

#endif //ESLAB_UTIL_MATH_HPP
