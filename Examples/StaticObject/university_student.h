#pragma once

#include <string>
#include "student.h"

class UniversityStudent : public Student {
  public:
    UniversityStudent();
    UniversityStudent(const std::string& name, const std::string& faculty);
    virtual ~UniversityStudent() = default;
    virtual void showDetails() override;
  
  private:
    int id_;
    static int id_counter_;
    std::string name_;
    std::string faculty_;
};
