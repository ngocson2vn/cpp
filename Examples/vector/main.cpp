#include <iostream>
#include <vector>

struct Student {
  int id = 0;
  std::string name;

  Student() = default;

  Student(int id, const std::string& name) : id(id), hidden_id_(id), name(name) {
    std::cout << "Student ctor with name = " << name << std::endl;
  };

  Student(Student&& src) noexcept {
    std::swap(id, src.id);
    name = std::move(src.name);

    // Keep src.hidden_id_ unchanged for tracking destruction
    hidden_id_ = src.hidden_id_;
    std::cout << "Student move ctor with name = " << name << std::endl;
  }

  void study() {
    std::cout << name << " is studying" << std::endl;
  }

  void dump() {
    std::cout << "Student id: " << id << ", name: " << name << std::endl;
  }

  ~Student() {
    std::cout << "Student dtor with id = " << hidden_id_ << std::endl;
  }

 private:
  int hidden_id_;
};

void printVector(const std::vector<int>& v) {
  for (auto& e : v) {
    std::cout << "Element: " << e << std::endl;
  }
}

int main() {
  std::vector<Student> students;
  for (int i = 0; i < 100; i++) {
    students.emplace_back(i, std::string("Student").append(std::to_string(i)));
  }
}