template <typename T>
struct foo {
  template <typename U>
  void bar() { }
};

template <typename T>
void func(foo<T> f) {
  f.bar<float>();
}

int main(int argv, char** argc) {
  return 0;
}