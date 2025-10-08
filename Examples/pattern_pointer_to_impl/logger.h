#pragma once

#include <memory>
#include <string>

// Forward declaration to avoid including heavy headers (e.g., <fstream>) here.
class Logger {
public:
    // Rule of five (we'll show a safe implementation in .cpp)
    Logger(const std::string& filePath);
    ~Logger();

    Logger(Logger&&) noexcept;
    Logger& operator=(Logger&&) noexcept;

    // Non-copyable (but you could implement deep copy if needed)
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    // Public API
    void setLevel(int level);
    void log(int level, const std::string& message);
    void flush();

private:
    // Private implementation (opaque pointer)
    class Impl;
    std::unique_ptr<Impl> impl_;  // or std::shared_ptr<Impl> or raw pointer
};
