//
// Created by pospelov on 18.12.2020.
//

#ifndef LINEAR_METHODS_HOUSEHOLDER_HPP_
#define LINEAR_METHODS_HOUSEHOLDER_HPP_

#include "../core/matrix.hpp"
#include "../core/util.hpp"

#include <optional>
#include <random>

namespace Linear {

extern const int ITERS;
extern const int LIMIT;
extern const double EPS;

template<typename T>
struct HouseholderMatrix {
  std::vector<T> v;
  explicit HouseholderMatrix(const std::vector<T> &v) : v(normalize(v)) {
  }
  explicit HouseholderMatrix(const std::vector<T> &v, int i) : HouseholderMatrix(normalize(v) - standart<T>(v.size(), i)) {

  }
};

template<typename T>
Matrix<T> operator *(const HouseholderMatrix<T> &H, const Matrix<T> &A) {
  Matrix<T> new_A = A - mult(H.v * 2.0, (A.transpose() * H.v));
  return new_A;
}


template<typename T>
std::pair<Matrix<T>, Matrix<T>> QR_householder(const Matrix<T> &A) {
  int n = A.n;
  Matrix<T> A0 = A;
  Matrix<T> Q = identity(n);
  for (int c = 0; c < n; c++) {
    std::vector<T> v(n);
    for (int i = c; i < n; i++) {
      v[i] = A0[i][c];
    }
    if (is_zero(v)) {
      continue;
    }

    HouseholderMatrix H(v, c);
    A0 = H * A0;
    Q = H * Q;
  }
  return {Q.transpose(), A0};
}

}

#endif//LINEAR_METHODS_HOUSEHOLDER_HPP_
