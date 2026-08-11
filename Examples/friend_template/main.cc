#include <iostream>
#include <memory>

namespace mlir {
class Pass {
public:
  Pass() { std::cout << "Pass obj: " << this << std::endl; }
};
} // namespace mlir

// This is a forward declaration of the friend function which will be defined
// inside `BarBase` class below. This is required because the public function
// `createBar()` looks for a qualified name `impl::createBar()`.
//
// Two crucial points:
// 1. If we omit this forward declaration, the compiler will fail to find a
// qualified name `impl::createBar()`.
//
// 2. By default, the friend function is hidden from both Unqualified Name
// Lookup and Qualified Name Lookup. However, since this forward declaration
// function exists, the friend function will later defines this forward
// declaration function.
//
// The Prior Declaration Rule
// If a function is already declared in the enclosing namespace, and you
// subsequently define it as an inline friend inside a class in that same
// namespace, the compiler binds the friend definition to your prior
// declaration. Because your prior declaration made the function visible to
// standard name lookups (both Unqualified Name Lookup and Qualified Name
// Lookup), it can be called normally without relying on Argument-Dependent
// Lookup (ADL).
//
// Therefore, the public function `createBar()` will eventually calls the friend
// function.
namespace impl {
std::unique_ptr<mlir::Pass> createBar();
}

namespace impl {

template <typename DerivedT> class BarBase : public mlir::Pass {
public:
  BarBase() { std::cout << "BarBase obj: " << this << std::endl; }

  friend std::unique_ptr<mlir::Pass> createBar() {
    return std::make_unique<DerivedT>();
  }
};

} // namespace impl

std::unique_ptr<mlir::Pass> createBar() {
  return impl::createBar();
}

class MyBar : public impl::BarBase<MyBar> {
public:
  MyBar() { std::cout << "MyBar obj: " << this << std::endl; }
};

int main(int argc, char **argv) { 
  auto bar = createBar(); 
}
