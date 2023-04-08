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

    MyClass(const MyClass& src) : cap(src.cap), data(src.data) {
      std::cout << "Executing MyClass Copy Constructor" << std::endl;
    }

    MyClass(MyClass&& rhs) {
      // Shallow copy
      cap = rhs.cap;
      data = rhs.data;
      
      // Reset rhs
      rhs.cap = 0;
      rhs.data = nullptr;
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

template <typename T>
struct strip_reference {
  typedef T type;
};

template <typename T>
struct strip_reference<T&> {
  typedef T type;
};

template <typename T>
struct strip_reference<T&&> {
  typedef T type;
};

template <class T>
typename strip_reference<T>::type&&
move(T&& x) {
  return static_cast<typename strip_reference<T>::type&&>(x);
}

int main() {
  MyClass m1(10);
  std::cout << "m1.data: " << m1.data << std::endl;

  MyClass m2 = move(m1);
  std::cout << "m2.data: " << m2.data << std::endl;

  std::cout << "\nRe-check m1" << std::endl;
  std::cout << "m1.data: " << m1.data << std::endl;
  for (int i = 0; i < m1.getCap(); i++) {
    std::cout << m1.data[i] << std::endl;
  }

  std::cout << "\nRe-check m2" << std::endl;
  std::cout << "m2.data: " << m2.data << std::endl << std::endl;
  for (int i = 0; i < m2.getCap(); i++) {
    std::cout << m2.data[i] << std::endl;
  }

  std::cout << "END" << std::endl;
}
