#include <iostream>

class MyClass {
  public: 
    std::string* data;
    friend void swap(MyClass& first, MyClass& second) noexcept;
    MyClass() = default;

    MyClass(int cap) {
      this->cap = cap;
      this->data = new std::string[cap];
      for (int i = 0; i < cap; i++) {
        data[i] = "DummyItem" + std::to_string(i);
      }
    }

    // Copy Constructor
    MyClass(const MyClass& src) {
      std::cout << "MyClass - Copy Constructor" << std::endl;
      cap = src.cap;
      data = new std::string[cap];
      for (int i = 0; i < cap; i++) {
        data[i] = src.data[i];
      }
    }

    // Delegate to default constructor
    // to initialize data members
    MyClass(MyClass&& src) noexcept : MyClass() {
      std::cout << "MyClass - Move Constructor" << std::endl;
      swap(*this, src);
    }

    MyClass& operator=(MyClass&& rhs) noexcept {
      std::cout << "MyClass - Move Assignment Operator" << std::endl;
      MyClass tmp(std::move(rhs));
      swap(*this, tmp);
      return *this;
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

void swap(MyClass& first, MyClass& second) noexcept {
  std::swap(first.cap, second.cap);
  std::swap(first.data, second.data);
}

void printMyClass(MyClass m) {
  std::cout << "The address of m.data: " << m.data << std::endl;
  for (int i = 0; i < m.getCap(); i++) {
    std::cout << m.data[i] << std::endl;
  }
}

int main() {
  std::vector<MyClass> payloads;
  std::cout << "payloads cap: " << payloads.capacity() << std::endl;
  payloads.push_back(MyClass(1));
  std::cout << "payloads cap: " << payloads.capacity() << std::endl;
  payloads.push_back(MyClass(2));
  std::cout << "payloads cap: " << payloads.capacity() << std::endl;
  payloads.push_back(MyClass(3));
  std::cout << "payloads cap: " << payloads.capacity() << std::endl;
  payloads.push_back(MyClass(4));
  std::cout << "payloads cap: " << payloads.capacity() << std::endl;

  MyClass m3(3);
  payloads.push_back(m3);

  std::cout << "END" << std::endl;
}
