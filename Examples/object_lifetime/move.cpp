#include <iostream>
#include <string>

struct Student {
    int id;
    std::string name;

    void study() {
        std::cout << name << " is studying" << std::endl;
    }

    ~Student() {
        std::cout << "~Student(): Goodbye!" << std::endl;
    }
};

int main(int argc, char** argv) {
    for (int i = 0; i < 10; i++) {
        std::cout << std::endl;
        Student student;
        student.id = i;
        student.name = "Student_" + std::to_string(i);
        student.study();
    }
}