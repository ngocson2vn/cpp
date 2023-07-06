#include <iostream>

class Simple{
  public:
    Simple() {
      counter++;
      std::cout << "[" << counter << "] " << "Simple constructor called!" << std::endl;
    }

    ~Simple() {
      std::cout << "[" << counter << "] " << "Simple destructor called!" << std::endl;
      counter--;
    }

  private:
    static int counter;
};

int ::Simple::counter = 0;

int main() {
  Simple* simpleArray = new Simple[5];
  std::cout << std::endl;

  delete[] simpleArray;
  simpleArray = nullptr;
}