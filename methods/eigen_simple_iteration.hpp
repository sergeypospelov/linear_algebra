//
// Created by pospelov on 13.12.2020.
//

#ifndef LINEAR_METHODS_EIGEN_SIMPLE_ITERATION_HPP_
#define LINEAR_METHODS_EIGEN_SIMPLE_ITERATION_HPP_

#include "../core/matrix.hpp"
#include "../core/util.hpp"

#include <optional>
#include <random>

namespace Linear {

extern const int LIMIT;

template<class T>
std::optional<std::pair<std::vector<T>, T>> eigen_simple_iteration(const Matrix<T> &A, const double EPS = 1e-3) {
  int n = A.n;

  auto v = normalize(random_vector<T>(n));

  int increase = 0;
  for (int iter = 0; iter < LIMIT; iter++) {
    v = normalize(A * v);
    T lambda = scalar(v, A * v);

    if (abs(A * v - v * lambda) < EPS) {
      return std::optional(make_pair(v, lambda));
    }
  }
  return std::nullopt;
}

};// namespace Linear

#endif//LINEAR_METHODS_SIMPLE_ITERATION_HPP_
