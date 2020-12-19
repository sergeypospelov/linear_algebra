//
// Created by pospelov on 13.12.2020.
//

#ifndef LINEAR_METHODS_SIMPLE_ITERATION_HPP_
#define LINEAR_METHODS_SIMPLE_ITERATION_HPP_

#include "../core/matrix.hpp"
#include "../core/util.hpp"

#include <optional>
#include <random>

namespace Linear {

const int ITERS = 20;
const int LIMIT = 1000;

template<class T>
std::optional<std::vector<T>> simple_iteration(const Matrix<T> &A, const std::vector<T> &b, const double EPS = 1e-3) {
  int n = A.n;
  if (!check_dimension(A, b)) {
    throw std::runtime_error("Bad arguments!");
  }

  auto circles = gershgorin_circles(A);
  long double rad = 0;
  for (auto &i : circles) {
    rad = std::max(rad, std::abs(i.first) + i.second);
  }

  bool bad_circles = rad >= 1;

  auto x = random_vector<T>(n);

  int increase = 0;
  long double prv_abs = abs(x);
  for (int iter = 0; iter < LIMIT; iter++) {
    x = A * x + b;
    long double cur_abs = abs(x);
    if (cur_abs >= prv_abs + 1) {
      increase++;
    } else {
      increase = 0;
    }
    prv_abs = cur_abs;

    if (abs(x - A * x - b) < EPS) {
      return std::optional(x);
    }
    if (increase >= ITERS && bad_circles) {
      return std::nullopt;
    }
  }
  return std::nullopt;
}

};// namespace Linear

#endif//LINEAR_METHODS_SIMPLE_ITERATION_HPP_
