#include <iostream>

typedef int Random(); // Random is a name of type "int()"

template <typename T>
class TypeDetector;

int main(int argc, char* argv[]) {
  TypeDetector<Random> td;

  return 0;
}
