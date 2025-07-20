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

int main(int argc, char** argv) {
    const Student s1{100, "Son"};
    set_score(&s1, 10);

    std::cout << std::endl;
    Student* student = new Student();
    display_type<decltype(*student)>();
    std::cout << std::endl;

    delete student;
}