#include "myutils/mtime.hpp"
#include <iostream>
#include <vector>
using namespace std;

int main() {

  const int max = 10000;
  vector<vector<double>> a(max,vector<double>(max));
  vector<double> x(max), y(max);

  auto res = mtime::mTime([&] {
    for (int j = 0; j < max; j++) {
      for (int i = 0; i < max; i++) {
        y[i] += a[i][j] * x[j];
      }
    }
  });

  cout << res << endl;
}

/*
{ 
0 {0,1,2,3}
1 {0,1,2,3}
2 {0,1,2,3}
3 {0,1,2,3}
}
*/
