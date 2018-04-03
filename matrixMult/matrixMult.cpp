#include "../myutils/mtime.hpp"
#include <iostream>
#include <random>
using namespace std;

template <int rows, int same, int cols>
int **matrixMult(int (&matA)[rows][same], int (&matB)[same][cols]) {
  int **res = new int *[rows];
  for (int i = 0; i < rows; i++)
    res[i] = new int[cols];

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      int tmp = 0;
      for (int k = 0; k < same; k++) {
        tmp += matA[i][k] * matB[k][j];
      }
      res[i][j] = tmp;
    }
  }
  return res;
}

int main() {
  int matA[1000][1000], matB[1000][1000];

  /*
  // Generate random matrices
  random_device seed;
  mt19937 rand(seed());
  uniform_int_distribution<int> dis(0, 10);
  for (auto &i : matA) {
    for (int &j : i) {
      j = dis(rand);
    }
  }

  for (auto &i : matB) {
    for (int &j : i) {
      j = dis(rand);
    }
  }
  */
  for (auto &i : matA) {
    for (int &j : i) {
      j = 1;
    }
  }

  for (auto &i : matB) {
    for (int &j : i) {
      j = 1;
    }
  }

  ///////////////////////////
  /*
    for (auto &i : matA) {
      for (int &j : i) {
        cout << j << " ";
      }
      cout << endl;
    }
    cout << "-------" << endl;

    for (auto &i : matB) {
      for (int &j : i) {
        cout << j << " ";
      }
      cout << endl;
    }
    */
  // cout << "-------" << endl;

  int **res = matrixMult(matA, matB);
  // auto time = mtime::mTime([&] { res = matrixMult(matA, matB); });

  // cout << "Time: " << time / 1000 << '\n';

  /*
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 5; j++) {
      cout << res[i][j] << " ";
    }
    cout << endl;
  }
  */
}
