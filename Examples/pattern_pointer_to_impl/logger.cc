#include "logger.h"

#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>

static constexpr std::size_t kMaxMessages = 1000;

class Logger::Impl {
 public:
  explicit Impl(const std::string& filePath)
    : level_(0),
      file_path_(filePath),
      out_(filePath, std::ios::app),
      msg_count_(0),
      file_count_(0) {
    if (!out_) {
        throw std::runtime_error("Failed to open log file: " + filePath);
    }
  }

  void setLevel(int level) { level_ = level; }

  void log(int level, const std::string& message) {
    if (level < level_) return;

    out_ << currentTimestamp() << " [L" << level << "] " << message << '\n';

    if (++msg_count_ == kMaxMessages) {
      out_ = std::ofstream(file_path_.append(".").append(std::to_string(++file_count_)));
    }
  }

  void flush() { out_.flush(); }

 private:
  int level_;
  std::ofstream out_;
  std::string file_path_;
  std::size_t msg_count_;
  std::size_t file_count_;

  static std::string currentTimestamp() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto tt = system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&tt, &tm);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
  }
};

// Public Logger methods simply delegate to Impl

Logger::Logger(const std::string& filePath) : impl_(new Impl(filePath)) {}

Logger::~Logger() = default; // unique_ptr<Impl> handles cleanup

Logger::Logger(Logger&& other) noexcept = default;

Logger& Logger::operator=(Logger&& other) noexcept = default;

void Logger::setLevel(int level) {
  impl_->setLevel(level);
}

void Logger::log(int level, const std::string& message) {
  impl_->log(level, message); 
}

void Logger::flush() { impl_->flush(); }
