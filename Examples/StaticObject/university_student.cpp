#include <iostream>
#include <memory>
#include "university_student.h"
#include "student_factory.h"

int UniversityStudent::id_counter_ = 0;

UniversityStudent::UniversityStudent() : id_(++id_counter_) {
  std::cout << "Default UniversityStudent constructor" << std::endl;
}

UniversityStudent::UniversityStudent(const std::string& name, const std::string& faculty) : id_(++id_counter_), name_(name), faculty_(faculty) {

}

void UniversityStudent::showDetails() {
  std::cout << "ID: " << id_ << std::endl;
  std::cout << "Name: " << name_ << std::endl;
  std::cout << "Faculty: " << faculty_ << std::endl;
}

class UniversityStudentFactory : public StudentFactory {
  public:
    UniversityStudentFactory() {
      std::cout << "UniversityStudentFactory Constructor" << std::endl;
    }

    virtual std::unique_ptr<Student> NewStudent(const std::string& name) override {
      return std::unique_ptr<Student>(new UniversityStudent(name, "DefaultFaculty"));
    }
};

class UniversityStudentRegistrar {
  public:
    UniversityStudentRegistrar() {
      std::cout << "UniversityStudentRegistrar Constructor" << std::endl;
      StudentFactory::Register("UNIVERSITY_STUDENT", std::shared_ptr<StudentFactory>(new UniversityStudentFactory()));
    }
};

static UniversityStudentRegistrar registrar;
