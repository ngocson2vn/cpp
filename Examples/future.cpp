#include <iostream>
#include <thread>
#include <future>

std::vector<int> fib(int n) {
  std::vector<int> values;
  int next = 0;
  for (int i = 0; i <= n; i++) {
    if (i < 2) {
      values.emplace_back(i);
    } else {
      next = values[i-2] + values[i-1];
      values.emplace_back(next);
    }
  }
  return values;
}

void fib2(int n, std::promise<std::vector<int>> p) {
  p.set_value(fib(n));
}

void printResult(const std::vector<int>& results) {
  std::cout << "Result: ";
  for (int i = 0; i < results.size(); i++) {
    if (i == 0) {
      std::cout << results[i];
    } else {
      std::cout << ", " << results[i];
    }
  }
  std::cout << std::endl;
  std::cout.flush();
}

int main() {
  // Shared variable
  int n = 10;
  std::vector<int> results;
  std::thread t([&results, n](){results = fib(n);});
  t.join();
  printResult(results);

  // Promise - Future pattern
  std::promise<std::vector<int>> p;
  std::future<std::vector<int>> f = p.get_future();
  n = 15;
  std::thread t2(fib2, n, std::move(p));
  results = f.get();
  t2.join();
  printResult(results);

  // Async
  n = 20;
  auto future = std::async(fib, n);
  results = future.get();
  printResult(results);
}
