#pragma once

#include <memory>
#include "student.h"

class StudentFactory {
  public:
    static void Register(const std::string& student_type, std::shared_ptr<StudentFactory> factory);
    static std::shared_ptr<StudentFactory> GetFactory(const std::string& student_type);

    virtual std::unique_ptr<Student> NewStudent(const std::string& name) {
      return nullptr;
    }
};