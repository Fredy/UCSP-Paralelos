#include "myutils/mtime.hpp"
#include <iostream>
#include <vector>
using namespace std;

int main() {
  const int max = 10000;
  
  vector<vector<double>> a(max,vector<double>(max));
  vector<double> x(max), y(max);

  auto res = mtime::mTime([&] {
    for (int i = 0; i < max; i++) {
      for (int j = 0; j < max; j++) {
        y[i] += a[i][j] * x[j];
      }
    }
  });

  cout << res << endl;
}
