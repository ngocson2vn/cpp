// Build this with c++17

#include <cstdio>
#include <tuple>

struct Base1 {
  void print() {
    printf("This is an empty struct!\n");
  }
};

struct Derived1 : Base1 {
  float m1;
  float m2;

  void print() {
    printf("Derived1: m1=%.2f m2=%.2f\n", m1, m2);
  }
};

struct Base2 {
  bool ok;
};

struct Derived2 : Base2 {
  float m1;
  float m2;

  void print() {
    printf("Derived2: ok=%s m1=%.2f m2=%.2f\n", ok ? "true" : "false", m1, m2);
  }
};

struct Base3 {
  bool ok;
  int id;
};

struct Derived3 : Base3 {
  std::tuple<float, float> data;

  void print() {
    printf("Derived3: ok=%s id=%d data={%.2f, %.2f}\n", ok ? "true" : "false", id, std::get<0>(data), std::get<1>(data));
  }
};

Derived1 build1(float a, float b) {
  // From c++17,
  //   - {} for creating Base subobject
  //   - a, b for m1, m2, respectively
  return {{}, a, b};
}

Derived2 build2(bool ok, float a, float b) {
  // From c++17,
  //   - ok for creating Base subobject
  //   - a, b for m1, m2, respectively
  return {ok, a, b};
}

Derived3 build3(bool ok, int id, float a, float b) {
  // From c++17,
  //   - ok, id for creating Base subobject
  //   - {a, b} for creating `data` member
  return {ok, id, {a, b}};
}

int main(int argc, char** argv) {
  Derived1 d1 = build1(0.1, 0.2);
  d1.print();

  Derived2 d2 = build2(true, 0.3, 0.4);
  d2.print();

  Derived3 d3 = build3(true, 1, 0.5, 0.6);
  d3.print();
}
