#include <chrono>
#include <utility>

namespace mtime {
// 10^6 microseconds = 1 second

// This functions measure the execution time of a function as microseconds.
template <typename Functor> inline auto mTime(Functor &&function) {
  using namespace std::chrono;
  using hrc = high_resolution_clock;

  hrc::time_point t1 = hrc::now();
  function();
  hrc::time_point t2 = hrc::now();
  auto duration = duration_cast<microseconds>(t2 - t1);
  return duration.count();
}

template <typename Functor> inline auto mAvgTime(int reps, Functor &&function) {
  long long res = 0;
  for (int i = 0; i < reps; i++)
    res += mTime(std::forward<Functor>(function));

  return res / reps;
}

template <typename Functor, typename InitFunc>
inline auto mAvgTimeWithInit(int reps, InitFunc &&initializer,
                             Functor &&function) {
  long long res = 0;
  for (int i = 0; i < reps; i++) {
    initializer();
    res += mTime(std::forward<Functor>(function));
  }
  return res / double (reps);
}

} // namespace time
