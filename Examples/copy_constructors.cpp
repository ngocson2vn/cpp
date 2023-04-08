#include <iostream>

class MyClass {
  public: 
    std::string* data;
    MyClass() = default;

    MyClass(int cap) {
      this->cap = cap;
      this->data = new std::string[cap];
      for (int i = 0; i < cap; i++) {
        data[i] = "DummyItem" + std::to_string(i);
      }
    }

    ~MyClass() {
      cap = 0;
      delete[] data;
      data = nullptr;
    }

    int getCap() {
      return cap;
    }

  private:
    int cap;
};

void printMyClass(MyClass m) {
  std::cout << "The address of m.data: " << m.data << std::endl;
  for (int i = 0; i < m.getCap(); i++) {
    std::cout << m.data[i] << std::endl;
  }
}

int main() {
  MyClass m1(10);
  std::cout << "m1.data: " << m1.data << std::endl;

  // DAMAGE m1
  // printMyClass(m1);

  MyClass m2(20);
  std::cout << "m2.data: " << m2.data << std::endl;

  // Shallow copy
  // m1.data will be orphaned
  m1 = m2;

  std::cout << "\nRe-check m1" << std::endl;
  std::cout << "m1.data: " << m1.data << std::endl;
  for (int i = 0; i < m1.getCap(); i++) {
    std::cout << m1.data[i] << std::endl;
  }

  std::cout << "END" << std::endl;
}
