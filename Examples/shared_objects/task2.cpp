#include "task2.h"

Task2::Task2() {
  std::cout << "Construct Task2" << std::endl;
}

void Task2::perform() {
  std::cout << "Perform Task2" << std::endl;
}

void perform() {
  Task1 task1;
  task1.perform();
  std::cout << "Perform Task2" << std::endl;
}
