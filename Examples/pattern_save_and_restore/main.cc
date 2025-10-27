#include <iostream>
#include <string>

struct EngineV1 {
  const std::string name = "V1";

  void start() {
    std::cout << "EngineV1 is started\n";
  }

  void stop() {
    std::cout << "EngineV1 is stopped\n";
  }
};

const EngineV1* getEngineV1() {
  return new EngineV1();
}

struct MyCar {
  std::string model;
  const EngineV1* engine;

  ~MyCar() {
    if (engine) {
      delete engine;
    }
  }
};

template <typename T>
void printType() {
  std::string func_name = __PRETTY_FUNCTION__;
  std::string tmp = func_name.substr(func_name.find_first_of("[") + 1);
  std::string type = tmp.substr(4, tmp.size() - 5);
  std::cout << type << std::endl;
}

template <typename T>
struct SaveAndRestore {
  SaveAndRestore(T& X) : X(X), OldValue(X) {}
  SaveAndRestore(T& X, const T& NewValue) : X(X), OldValue(X) { X = NewValue; }
  SaveAndRestore(T& X, T&& NewValue) : X(X), OldValue(std::move(X)) {
    std::cout << "X has type = "; printType<decltype(X)>();
    std::cout << "NewValue has type = "; printType<decltype(NewValue)>();
    X = std::move(NewValue);
  }
  ~SaveAndRestore() { X = std::move(OldValue); }
  const T& get() { return OldValue; }

private:
  T& X;
  T OldValue;
};

int main() {
  MyCar c{"PT", nullptr};
  std::cout << "c.engine has type = "; printType<decltype(c.engine)>();
  std::cout << "c.engine = " << c.engine << std::endl;

  {
    SaveAndRestore engineGuard(c.engine, getEngineV1());
    std::cout << "c.engine = " << c.engine << std::endl;
  }

  std::cout << "c.engine = " << c.engine << std::endl;
}