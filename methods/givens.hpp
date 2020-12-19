//
// Created by pospelov on 17.12.2020.
//

#ifndef LINEAR_METHODS_GIVENS_HPP_
#define LINEAR_METHODS_GIVENS_HPP_

#include "../core/matrix.hpp"
#include "../core/util.hpp"

#include <optional>
#include <random>

namespace Linear {

extern const double EPS;

struct GivensMatrix {
  int i, j;
  double c, s;
  GivensMatrix(int i, int j, double c, double s) : i(i), j(j), c(c / hypot(c, s)), s(s / hypot(c, s)) {
  }
  GivensMatrix() : i(0), j(0), c(0), s(1) {
  }
  GivensMatrix inverse() const {
    return GivensMatrix(i, j, c, -s);
  }
};

template<typename T>
Matrix<T> operator *(const GivensMatrix &G, const Matrix<T> &A) {
  Matrix<T> new_A = A;
  new_A[G.i] = A[G.i] * G.c + A[G.j] * G.s;
  new_A[G.j] = -A[G.i] * G.s + A[G.j] * G.c;
  return new_A;
}


template<typename T>
std::pair<Matrix<T>, Matrix<T>> QR_givens(const Matrix<T> &A) {
  int n = A.n;
  Matrix<T> A0 = A;
  Matrix<T> Q = identity(n);
  for (int c = 0; c < n; c++) {
    int r = c;
    while (r < n && is_zero(A0[r][c])) {
      r++;
    }
    for (int i = r + 1; i < n; i++) {
      GivensMatrix G(r, i, A0[r][c], A0[i][c]);
      A0 = G * A0;
      Q = G * Q;
    }
    GivensMatrix G(r, r, 0, 1);
    A0 = G * A0;
    Q = G * Q;
  }
  return {Q.transpose(), A0};
}

}

#endif//LINEAR_METHODS_GIVENS_HPP_
