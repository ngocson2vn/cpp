#include <cstdio>
#include <iostream>
#include <string>
#include <initializer_list>


// The primary template serves as the base case for recursion (when no types remain in the pack).
template<int... Ns>
struct Arith {
  static int sum() {
    return 0;
  }
};

/* 
This is a partial specialization of the primary template. 
It is more specific because it constrains the parameter pack Ns... from the primary template to a non-empty pack, where:
- The first type in the pack is explicitly named N0.
- The remaining types are captured as Ns... (another variadic pack).

This specialization matches when there is at least one type in the parameter pack (i.e., N0, Ns...).
*/
template<int N0, int... Ns>
struct Arith<N0, Ns...> {
  static int sum() {
    return N0 + Arith<Ns...>::sum();
  }
};

int main() {
  int total = Arith<1, 2, 3>::sum();
  std::cout << "total = " << total << std::endl;
}
