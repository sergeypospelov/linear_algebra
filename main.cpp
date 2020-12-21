#include "core/matrix.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>

#include "methods/eigen_qr.hpp"
#include "methods/eigen_qr_shifts.hpp"
#include "methods/eigen_simple_iteration.hpp"
#include "methods/givens.hpp"
#include "methods/householder.hpp"
#include "methods/seidel.hpp"
#include "methods/simple_iteration.hpp"
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

void task4_3() {
  Matrix A({{0., 0, 0, 1, 1},
            {0., 2, 0, 0, 0},
            {1., 4, 4, 5, 7},
            {6., 0, 0, 0, 0},
            {0., 0, 1, 5, 9}});
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

void task10_1() {
  Matrix A({{1., 3, 3, 7},
            {3, 4, 0, 9},
            {3, 0, 0, 6},
            {7, 9, 6, 9}});
  auto res = tridiagonalization(A);
  A = res.first; // tridiagonalized
  cout << "tridiagonalized:\n";
  cout << A << "\n";

  auto res2 = eigen_qr_tridiagonalization(A);
  if (res2.has_value()) {
    cout << "eigen values:\n"
         << res2.value().first << "\n";
    cout << "eigen vectors:\n"
         << res2.value().second << "\n";
  } else {
    cout << ":(\n";
  }
}

void task10_2() {
  Matrix A({{1., 2, 3, 4, 5},
            {2, 2, 9, 16, 25},
            {3, 9, 16, 64, 125},
            {4, 16, 64, 256, 625},
            {5, 25, 125, 625, 3125}});
  auto res = tridiagonalization(A);
  A = res.first; // tridiagonalized
  cout << "tridiagonalized:\n";
  cout << A << "\n";

  auto res2 = eigen_qr_tridiagonalization(A);
  if (res2.has_value()) {
    cout << "eigen values:\n"
         << res2.value().first << "\n";
    cout << "eigen vectors:\n"
         << res2.value().second << "\n";
  } else {
    cout << ":(\n";
  }

}

/*
 * Graph isomorphism test!
 */

void task12_1() {
  int n = 7;
  vector<pair<int, int>> edges1 = {{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 1}}; // cycle
  Matrix G1(n);
  for (auto [u, v] : edges1) {
    G1[u - 1][v - 1] = G1[v - 1][u - 1] = 1;
  }

  vector<int> perm(n);
  for (int i = 0; i < n; i++) {
    perm[i] = i;
  }
  shuffle(perm.begin(), perm.end(), mt19937());

  Matrix G2(n);
  for (auto [u, v] : edges1) {
    int pu = perm[u - 1], pv = perm[v - 1];
    G2[pu][pv] = G2[pv][pu] = 1;
  }

  vector eig_values1 = eigen_qr_shift(tridiagonalization(G1).first).value().first;
  sort(eig_values1.begin(), eig_values1.end());

  vector eig_values2 = eigen_qr_shift(tridiagonalization(G2).first).value().first;
  sort(eig_values2.begin(), eig_values2.end());

  bool possible_isomorphic = 1;
  for (int i = 0; i < n; i++) {
    possible_isomorphic &= is_zero(eig_values1[i] - eig_values2[i]);
  }

  cout << (possible_isomorphic ? "maybe" : "not") << "\n";
}

void task12_2() {
  int n = 7;
  vector<pair<int, int>> edges1 = {{1, 2}, {4,3}, {1, 4}, {6, 5}, {1, 6}, {1, 7}};
  Matrix G1(n);
  for (auto [u, v] : edges1) {
    G1[u - 1][v - 1] = G1[v - 1][u - 1] = 1;
  }

  vector<pair<int, int>> edges2 = {{1, 2}, {1,3}, {1, 4}, {1, 5}, {1, 6}, {6, 7}}; // star with long path
  Matrix G2(n);
  for (auto [u, v] : edges2) {
    G2[u - 1][v - 1] = G2[v - 1][u - 1] = 1;
  }

  vector eig_values1 = eigen_qr_shift(tridiagonalization(G1).first).value().first;
  sort(eig_values1.begin(), eig_values1.end());

  vector eig_values2 = eigen_qr_shift(tridiagonalization(G2).first).value().first;
  sort(eig_values2.begin(), eig_values2.end());

  bool possible_isomorphic = 1;
  for (int i = 0; i < n; i++) {
    possible_isomorphic &= is_zero(eig_values1[i] - eig_values2[i]);
  }

  cout << (possible_isomorphic ? "maybe" : "not") << "\n";
}

void task12_0() {
  Matrix A({{0, 1.0}, {1.0, 0.0}});
  auto res = eigen_qr_shift(A);
  cout << res.value().first << " " << res.value().second << "\n";
}


void task13_1(int n) {
  Matrix<double> G(n * n);

  auto add = [&n, &G](int x, int y, int x2, int y2) {
    x = (x % n + n) % n;
    x2 = (x % n + n) % n;
    y = (y % n + n) % n;
    y2 = (y2 % n + n) % n;
    G[x * n + y][x2 * n + y2]++;
  };


  for (int x = 0; x < n; x++) {
    for (int y = 0; y < n; y++) {
      add(x, y, x + 2 * y, y);
      add(x, y, x - 2 * y, y);
      add(x, y, x + 2 * y + 1, y);
      add(x, y, x - 2 * y - 1, y);
      add(x, y, x, y + 2 * x);
      add(x, y, x, y - 2 * x);
      add(x, y, x, y + 2 * x + 1);
      add(x, y, x, y - 2 * x - 1);
    }
  }

  vector eig_values = eigen_qr_shift(tridiagonalization(G).first, 0).value().first;
  sort(eig_values.rbegin(), eig_values.rend());
  cout << "eigen values:\n" << eig_values << "\n";

  cout << "alpha = " << std::max(std::abs(eig_values[1]), std::abs(eig_values.back())) / 8 << "\n";
}

void task13_2(int p) { // p should be prime-number
  Matrix<double> G(p + 1);
  for (int i = 0; i < p; i++) {
    if (i > 0) {
      for (int j = 0; j < p; j++) {
        if (i * j % p == 1) {
          G[i][j]++;
          break;
        }
      }
    } else {
      G[i][p]++;
    }
    G[i][(i + 1) % p]++;
    G[i][(i - 1 + p) % p]++;
  }
  G[p][0] = 1;
  G[p][p] = 2;
  ofstream f_out;
  f_out.open("matrix");
  f_out << G;

  vector eig_values = eigen_qr_shift(tridiagonalization(G).first).value().first;
  sort(eig_values.rbegin(), eig_values.rend());
  cout << "eigen values:\n" << eig_values << "\n";

  cout << "alpha = " << std::max(std::abs(eig_values[1]), std::abs(eig_values.back())) / 3 << "\n";
}

int main() {
  cerr << fixed << setprecision(3);
//  task1();
//  task2();
//  task3();
//  task4_1();
//  task4_2();
//  task4_3();
//  task6_1();
//  task6_2();
//  task7();
//  task8();
//  task9_1();
//  task9_2();
//  task10_1();
//  task10_2();

//  task12_1();
//  task12_2();
//  task12_0();

//  task13_1(10); // takes about 5 seconds for n = 10 and about 20 secs for n = 15
//  task13_1(15);
//  task13_1(20); // on my PC it took 3 minutes
  task13_2(239);

  return 0;
}
