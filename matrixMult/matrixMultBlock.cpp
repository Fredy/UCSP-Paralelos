#include "../myutils/mtime.hpp"
#include <algorithm>
#include <iostream>
#include <random>
using namespace std;

template <int size>
int **matrixMult(int (&matA)[size][size], int (&matB)[size][size]) {
  int **res = new int *[size];
  for (int i = 0; i < size; i++)
    res[i] = new int[size];

  int block_size = 64 / sizeof(int); // Usally cache line is 64 bytes so 64 /
                                     // sizeof(int) integers can be there
  int temp;
  for (int jj = 0; jj < size; jj += block_size) {
    for (int kk = 0; kk < size; kk += block_size) {
      for (int i = 0; i < size; i++) {
        for (int j = jj;
             j < ((jj + block_size) > size ? size : (jj + block_size)); j++) {
          temp = 0;
          for (int k = kk;
               k < ((kk + block_size) > size ? size : (kk + block_size)); k++) {
            temp += matA[i][k] * matB[k][j];
          }
          res[i][j] += temp;
        }
      }
    }
  }
  return res;
}

int main() {
  int matA[1000][1000], matB[1000][1000];

  // Generate random matrices
  /*
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

  //  cout << "Time: " << time / 1000.0 << '\n';

  /*
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 5; j++) {
      cout << res[i][j] << " ";
    }
    cout << endl;
  }
  */
}
