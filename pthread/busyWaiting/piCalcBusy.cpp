#include "../myutils/mtime.hpp"
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>
using namespace std;

constexpr int MAX_THREADS = 1024;

void threadSum(const int rank, const int threadCount, const size_t n,
               volatile double &sum, volatile int &flag) {
  double factor, mySum = 0.0;
  size_t myN = n / threadCount;
  size_t myFirstI = myN * rank;
  size_t myLastI = myFirstI + myN;

  if (myFirstI % 2 == 0) {
    factor = 1.0;
  } else {
    factor = -1.0;
  }

  for (size_t i = myFirstI; i < myLastI; i++, factor = -factor) {
    mySum += factor / (2 * i + 1);
  }

  while (flag != rank); // Busy waiting
  sum += mySum;
  flag = (flag + 1) % threadCount; // Busy waiting
}

void usage(char *prog_name) {
  cerr << "usage: " << prog_name << " <number of threads> <n>\n";
  cerr << "   n is the number of terms and should be >= 1\n";
  cerr << "   n should be evenly divisible by the number of threads\n";
  exit(0);
}

void getArgs(int argc, char *argv[], size_t &threadCount, size_t &n) {
  if (argc != 3)
    usage(argv[0]);
  threadCount = stoull(argv[1]);
  if (threadCount <= 0 or threadCount > MAX_THREADS)
    usage(argv[0]);
  n = stoull(argv[2]);
  if (n <= 0)
    usage(argv[0]);
}

int main(int argc, char *argv[]) {
  size_t threadCount;
  size_t n;
  volatile int flag = 0;
  volatile double sum = 0.0;

  getArgs(argc, argv, threadCount, n);

  vector<thread> threadHandles(threadCount);

  auto totalTime = mtime::mTime([&] {
    for (size_t i = 0; i < threadCount; i++) {
      threadHandles[i] =
          thread(threadSum, i, threadCount, n, ref(sum), ref(flag));
    }

    for (auto &threadHandle : threadHandles) {
      threadHandle.join();
    }

    sum = 4.0 * sum;
  });

  cout << scientific << totalTime / 1000.0 << '\n';
  //  cout << resetiosflags(ios::scientific) << setprecision(15) << sum << '\n';
}
