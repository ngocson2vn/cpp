template <typename T>
struct type_or_value;

template <>
struct type_or_value<int> {
  static const bool tv = true;
};

template <>
struct type_or_value<float> {
  using tv = float;
};

template <typename T>
void func() {
  using t = typename type_or_value<T>::tv;
  bool v = type_or_value<T>::tv;
}

int main(int argc, char** argv) {
  return 0;
}
