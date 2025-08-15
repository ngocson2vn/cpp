#include <iostream>
#include <filesystem>
#include <fstream>

struct Student {
    int id;
    std::string name;

    Student() = default;

    Student(int id, const std::string& name) : id(id), name(name) {}

    void study() const {
        std::cout << name << " is studying" << std::endl;
    }

    ~Student() {
        std::cout << "~Student(): Goodbye!" << std::endl;
    }
};

void check(const Student& s) {
  s.study();
}

int main() {
  Student s(10, "Sony");
  check(s);
  return 0;
}