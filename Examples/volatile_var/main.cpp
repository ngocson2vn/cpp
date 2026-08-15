#include <csignal>
#include <iostream>

// The volatile keyword tells the compiler that this variable
// can be changed by something outside the current thread of
// execution (another thread, hardware, signal handler, etc.).
// Therefore the compiler must not optimize away repeated reads
// or writes of this variable.
// Change to plain "bool running = true;" to see the optimization
bool running = true;
// volatile bool running = true;

void sigterm_handler(int sig) { 
  std::cout << "Received SIGTERM (" << sig << ")\n";
  running = false; 
}

int main() {
  std::signal(SIGTERM, sigterm_handler);

  std::cout << "Waiting for to be signaled...\n";

  // Without 'volatile', a sufficiently aggressive optimizer
  // could turn this loop into an infinite loop, because it
  // would assume that 'ready' never changes inside the loop.
  int temp = 0;
  int prev = 0;
  int next = 1;
  int result = 0;
  int N = 1000;
  while (running) {
    for (int i = 0; i < N; i++) {
      temp = prev + next;
      prev = next;
      next = temp;
    }

    // The compiler thinks that `result` is never observable because the while loop run forever.
    result = next;
  }

  std::cout << "Result: " << result << "\n";

  std::cout << "DONE!\n";

  return 0;
}
