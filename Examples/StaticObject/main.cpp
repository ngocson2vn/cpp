#include <iostream>
#include <memory>
#include "student_factory.h"

static const std::string STUDENT_TYPE = "UNIVERSITY_STUDENT";

int main() {
  std::cout << "\nStart main\n" << std::endl;
  std::shared_ptr<StudentFactory> factory = StudentFactory::GetFactory(STUDENT_TYPE);
  if (!factory) {
    std::cout << "Failed to get a factory for " << STUDENT_TYPE << std::endl;
    return 1;
  }
  
  std::string name("DummyName");
  std::unique_ptr<Student> student;
  for (int i = 0; i < 3; i++) {
    student = factory->NewStudent(name + std::to_string(i+1));
    student->showDetails();
    std::cout << std::endl;
  }

  std::cout << "\nEnd main" << std::endl;
}
