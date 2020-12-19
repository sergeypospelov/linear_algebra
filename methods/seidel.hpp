//
// Created by pospelov on 16.12.2020.
//

#ifndef LINEAR_METHODS_SEIDEL_HPP_
#define LINEAR_METHODS_SEIDEL_HPP_

#include "../core/matrix.hpp"
#include "../core/util.hpp"

#include <optional>
#include <random>

namespace Linear {

extern const int ITERS;
extern const int LIMIT;

template<typename T>
std::pair<Matrix<T>, Matrix<T>> get_LU(const Matrix<T> &A) {
  Matrix<T> L(A.n), U(A.n);
  for (int i = 0; i < A.n; i++) {
    for (int j = 0; j < A.n; j++) {
      if (i >= j) {
        L[i][j] = A[i][j];
      } else {
        U[i][j] = A[i][j];
      }
    }
  }
  return {L, U};
}

template<typename T>
std::vector<T> solve_LU(const Matrix<T> &L, const Matrix<T> &U, const std::vector<T> &b, const std::vector<T> &prev_x) {
  int n = L.n;
  auto sum = -U * prev_x + b;
  std::vector<T> new_x(n);
  for (int i = 0; i < n; i++) {
    new_x[i] = sum[i] / L[i][i];
    for (int j = i + 1; j < n; j++) {
      T k = L[j][i] / L[i][i];
      sum[j] -= sum[i] * k;
    }
  }
  return new_x;
}

template<typename T>
std::optional<std::vector<T>> seidel(const Matrix<T> &A, const std::vector<T> &b, const double EPS = 1e-3) {
  int n = A.n;

  if (!check_dimension(A, b)) {
    throw std::runtime_error("Bad arguments!");
  }

  for (int i = 0; i < n; i++) {
    if (A[i][i] == 0) {
      throw std::runtime_error("Bad arguments!");
    }
  }

  auto [L, U] = get_LU(A);
  auto x = random_vector<T>(n);

  int increase = 0;
  long double prv_abs = abs(x);
  for (int iter = 0; iter < LIMIT; iter++) {
    x = solve_LU(L, U, b, x);
    long double cur_abs = abs(x);
    if (cur_abs >= prv_abs + 1) {
      increase++;
    } else {
      increase = 0;
    }
    prv_abs = cur_abs;

    if (abs(A * x - b) < EPS) {
      return std::optional(x);
    }
    if (increase >= ITERS) {
      return std::nullopt;
    }
  }
  return std::nullopt;
}

}

#endif//LINEAR_METHODS_SEIDEL_HPP_
