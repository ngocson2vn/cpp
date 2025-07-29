#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <unordered_set>
#include <sstream>
#include <array>

// C APIs
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <errno.h>

std::vector<std::string> splitStringBySpace(const std::string& str) {
  std::vector<std::string> tokens;
  std::stringstream ss(str); // Initialize stringstream with the input string
  std::string token;

  // Extract words (tokens) from the stringstream until no more words are found
  while (ss >> token) { 
    tokens.push_back(token); // Add the extracted token to the vector
  }
  return tokens;
}

void runCommand(const std::string& cmd, std::string& stdoutOutput, std::string& stderrOutput, bool& status) {
  std::vector<std::string> parts = splitStringBySpace(cmd);
  std::string prog = parts[0];
  int numArgs = parts.size();
  std::unique_ptr<char*> args_ptr(new char*[numArgs]);
  char** argv = args_ptr.get();
  for (int i = 0; i < numArgs - 1; i++) {
    argv[i] = parts[i + 1].data();
  }
  argv[numArgs - 1] = nullptr;

  std::cout << "prog: " << prog << std::endl;
  for (int i = 0; i < numArgs - 1; i++) {
    std::cout << "argv[" << i << "]: " << argv[i] << std::endl;
  }

  int stdoutPipe[2], stderrPipe[2];
  if (pipe(stdoutPipe) == -1 || pipe(stderrPipe) == -1) {
    status = false;
    std::cerr << "[runCommand] Failed to open stdout/stderr pipe" << std::endl;
    return;
  }

  pid_t pid = fork();
  if (pid == -1) {
    status = false;
    std::cerr << "[runCommand] Failed to fork this program" << std::endl;
    return;
  }

  // Child process
  if (pid == 0) {
    // Close read ends of pipes
    close(stdoutPipe[0]);
    close(stderrPipe[0]);

    // Redirect stdout and stderr to respective pipes
    dup2(stdoutPipe[1], STDOUT_FILENO);
    dup2(stderrPipe[1], STDERR_FILENO);
    close(stdoutPipe[1]);
    close(stderrPipe[1]);

    // Execute command
    execvp(prog.c_str(), argv);

    // execvp only returns if an error occurred
    fprintf(stderr, "[runCommand] execvp %s: %s\n", prog.c_str(), strerror(errno));
    exit(EXIT_FAILURE); // Child process exits
  }

  // Parent process
  else {
    // Close write ends of pipes
    close(stdoutPipe[1]);
    close(stderrPipe[1]);

    // Read from pipes
    std::array<char, 128> buffer;
    ssize_t bytesRead;

    // Read stdout
    while ((bytesRead = read(stdoutPipe[0], buffer.data(), buffer.size() - 1)) > 0) {
      buffer[bytesRead] = '\0';
      stdoutOutput += buffer.data();
    }
    close(stdoutPipe[0]);

    // Read stderr
    while ((bytesRead = read(stderrPipe[0], buffer.data(), buffer.size() - 1)) > 0) {
      buffer[bytesRead] = '\0';
      stderrOutput += buffer.data();
    }
    close(stderrPipe[0]);

    // Wait for child to finish
    int ret;
    waitpid(pid, &ret, 0);
    status = (WEXITSTATUS(ret) == 0);
  }
}

int main(int argc, char** argv) {
  std::string stdoutOutput, stderrOutput;
  bool ok;
  runCommand("/usr/bin/ls -l -r -t /data00/home/son.nguyen/workspace/learnmlir/study_triton_compiler", stdoutOutput, stderrOutput, ok);
  if (ok) {
    std::cout << "\nstdout:\n" << stdoutOutput << std::endl;
  } else {
    std::cerr << "Command execution failed\n\n";
    std::cerr << "stderr:\n" << stderrOutput << std::endl;
    return 1;
  }

  return 0;
}