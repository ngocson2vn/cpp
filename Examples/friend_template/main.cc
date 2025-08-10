#include <iostream>
#include <memory>

namespace mlir {
class Pass {
 public:
  Pass() {
    std::cout << "Pass obj: " << this << std::endl;
  } 
};
}

namespace impl {
  std::unique_ptr<mlir::Pass> createBar();
}

namespace impl {

template <typename DerivedT>
class BarBase : public mlir::Pass {
 public:
  BarBase() {
    std::cout << "BarBase obj: " << this << std::endl;
  }

  friend std::unique_ptr<mlir::Pass> createBar() {
    std::cout << __PRETTY_FUNCTION__ << std::endl;
    return std::make_unique<DerivedT>();
  }
};

} // namespace impl


std::unique_ptr<mlir::Pass> createBar() {
  return impl::createBar();
}

class MyBar : public impl::BarBase<MyBar> {
 public:
  MyBar() {
    std::cout << "MyBar obj: " << this << std::endl;
  }
};

int main(int argc, char** argv) {
  auto bar = createBar();
}
