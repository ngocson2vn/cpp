#include <iostream>
#include <string>

struct Student {
    int id;
    std::string name;

    Student() = default;

    Student(int id, const std::string& name) : id(id), name(name) {}

    void study() {
        std::cout << name << " is studying" << std::endl;
    }

    ~Student() {
        std::cout << "~Student(): Goodbye!" << std::endl;
    }
};

struct Score {
    const Student* student;
    int point;
};

void save_score_record(Score score) {
    std::cout << "student: " << score.student << std::endl;
    std::cout << "Save score record for student " << (score.student)->name << std::endl;
}

void set_score(const Student* student, int p) {
    std::cout << "student: " << student << std::endl;
    save_score_record({student, p});
}

template <typename T>
void display_type() {
  std::string func_name(__PRETTY_FUNCTION__);
  std::string tmp = func_name.substr(func_name.find_first_of("[") + 1);
  std::string type = "type" + tmp.substr(1, tmp.size() - 2);
  std::cout << type << std::endl;
}

typedef unsigned long long PtrType;

int main(int argc, char** argv) {
    const Student s1{100, "Son"};
    set_score(&s1, 10);

    std::cout << std::endl;

    // student holds a memory address (a 64-bit integer number)
    Student* student = new Student(10, "Foo");

    display_type<decltype(*student)>();
    std::cout << std::endl;

    // Explicitly cast the memory address that student holds to an `unsigned long long` number
    PtrType studentPtr = reinterpret_cast<PtrType>(student);
    std::cout << "student: " << student << std::endl;
    std::cout << "studentPtr: " << studentPtr << std::endl;
    Student* sptr = reinterpret_cast<Student*>(studentPtr);
    sptr->study();

    void* x;
    void** xPtr = &x;
    
    // Copy memory address to x
    *xPtr = reinterpret_cast<void*>(student);
    std::cout << "x: " << x << std::endl;

    delete student;
}