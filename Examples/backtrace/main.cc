#include <iostream>
#include <execinfo.h>
#include <unistd.h>

void printBacktrace() {
    const int max_frames = 64;
    void* buffer[max_frames];
    
    // Get the return addresses
    int num_frames = backtrace(buffer, max_frames);
    
    // Print them out to standard error
    std::cerr << "Backtrace:\n";
    backtrace_symbols_fd(buffer, num_frames, STDERR_FILENO);
}

void myFunction() {
    printBacktrace();
}

int main() {
    myFunction();
    return 0;
}