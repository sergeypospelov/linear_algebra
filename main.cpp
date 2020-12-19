#include "core/matrix.hpp"
#include <iostream>

#include "methods/seidel.hpp"
#include "methods/simple_iteration.hpp"
#include "methods/givens.hpp"
#include "methods/eigen_simple_iteration.hpp"
#include "methods/householder.hpp"
#include "methods/eigen_qr.hpp"
#include "methods/tridiagonalization.hpp"

using namespace Linear;
using namespace std;

/*
 * Simple iteration method.
 */
void task1() {
  Matrix a({{0.3, 0, 0},
            {0, 0.4, 0},
            {0.5, 0.5, 0.5}});
  vector b = {1.0, 2.0, 3.0};
  auto res = simple_iteration(a, b);

  if (res.has_value()) {
    cout << res.value() << "\n"
         << a * res.value() + b << "\n";
  } else {
    cout << ":(";
  }
}

/*
 * Gauss-Seidel method.
 */
void task2() {
  Matrix a({{10.0, -1, 2, 0},
            {-1, 11, -1, 3},
            {2, -1, 10, -1},
            {0, 3, -1, 8}});
  /*
  Matrix a({{16.0, 3},
            {7, -11}});
  vector b = {11.0, 13.0};
*/
  vector b = {6.0, 25.0, -11.0, 15.0};
  auto res = seidel(a, b);

  if (res.has_value()) {
    cout << res.value() << "\n"
         << a * res.value() - b << "\n";
  } else {
    cout << ":(";
  }
}

/*
 * One Givens rotation
 */

void task3() {
  Matrix A({{6., 5., 0.},
            {5., 1., 4.},
            {0., 4., 3.}});
  GivensMatrix G(0, 1, 6, 5);
  cout << G * A << "\n";
}

/*
 * QR algorithm with Givens rotations.
 */

void task4_1() {
  Matrix A({{6., 5., 0.},
            {5., 1., 4.},
            {0., 4., 3.}});
  auto res = QR_givens(A);
  cout << "Q =\n" << res.first << "R = \n";
  cout << res.second << "\n";
  cout << "A - QR = \n" << (A - res.first * res.second) << "\n";
}

void task4_2() {
  Matrix A({{1., 3, 3, 7},
            {2, 4, 0, 9},
            {1, 3, 0, 6},
            {6, 9, 6, 9}});
  auto res = QR_givens(A);
  cout << "Q =\n" << res.first << "R = \n";
  cout << res.second << "\n";
  cout << "A - QR = \n" << (A - res.first * res.second) << "\n";
}

void task5() {
  Matrix A({{1., 3, 3, 7},
            {2, 4, 0, 9},
            {1, 3, 0, 6},
            {6, 9, 6, 9}});
  HouseholderMatrix<double> H({0, 1, 1, 0});
  cout << H * A << "\n";
}

/*
 * Householder QR algorithm
 */

void task6_1() {
  Matrix A({{6., 5., 0.},
            {5., 1., 4.},
            {0., 4., 3.}});
  auto res = QR_householder(A);
  cout << "Q =\n" << res.first << "R = \n";
  cout << res.second << "\n";
  cout << "A - QR = \n" << (A - res.first * res.second) << "\n";

}

void task6_2() {
  Matrix A({{1., 3, 3, 7},
            {2, 4, 0, 9},
            {1, 3, 0, 6},
            {6, 9, 6, 9}});
  auto res = QR_householder(A);
  cout << "Q =\n" << res.first << "R = \n";
  cout << res.second << "\n";
  cout << "A - QR = \n" << (A - res.first * res.second) << "\n";
}

/*
 * Simple iteration method for eigen values
 */

void task7() {
  Matrix A({{0.3, 0.2, 0},
            {0.4, 0.7, 0.1},
            {0.5, 0.5, 0.5}});
  auto res = eigen_simple_iteration(A);

  if (res.has_value()) {
    cout << "Eigen vector:\n" << res.value().first << "\n"
         << "Eigen value: " << res.value().second << "\n\n";
  } else {
    cout << ":(\n";
  }
}

/*
 * Computation of eigen values and vectors using QR-algorithm
 */

void task8() {
  Matrix A({{1., 3, 3, 7},
            {3, 4, 0, 9},
            {3, 0, 0, 6},
            {7, 9, 6, 9}});

  auto res = eigen_qr(A);
  if (res.has_value()) {
    cout << "eigen values:\n"
         << res.value().first << "\n";
    cout << "eigen vectors:\n"
         << res.value().second << "\n";
  } else {
    cout << ":(\n";
  }
}

/*
 * Tridiagonalization.
 */

void task9_1() {
  Matrix A({{1., 3, 3, 7},
            {3, 4, 0, 9},
            {3, 0, 0, 6},
            {7, 9, 6, 9}});
  auto res = tridiagonalization(A);
  cout << "A' =\n" << res.first << "\n";
  cout << "Q =\n" << res.second << "\n";
  cout << "Q^T * A * Q - A' =\n" << res.second.transpose() * A * res.second - res.first << "\n";
}

void task9_2() {
  Matrix A({{1., 2, 3, 4, 5},
            {2, 2, 9, 16, 25},
            {3, 9, 16, 64, 125},
            {4, 16, 64, 256, 625},
            {5, 25, 125, 625, 3125}});
  auto res = tridiagonalization(A);
  cout << "A' =\n" << res.first << "\n";
  cout << "Q =\n" << res.second << "\n";
  cout << "Q^T * A * Q - A' =\n" << res.second.transpose() * A * res.second - res.first << "\n";
}



int main() {
  //  task1();
  //  task2();
  //  task3();
  //  task4_1();
  //  task4_2();
  //  task6_1();
  //  task6_2();
  //  task7();
  //  task8();
    task9_1();
    task9_2();

  return 0;
}
