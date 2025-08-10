#include <iostream>
#include <string>

struct Student {
    int id;
    std::string name;

    Student() = default;

    Student(int id, const std::string& name = "")
        : id(id), name(name) {};

    void study() {
        std::cout << name << " is studying" << std::endl;
    }

    void dump() {
        std::cout << "Student id: " << id << ", name: " << name << std::endl;
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

Student get_student_1000() {
    return 1000;
}

Student get_student_3000() {
    return Student(3000, "Student3000");
}

int main(int argc, char** argv) {
    Student s;
    s.dump();

    const Student s1{100, "Son"};
    set_score(&s1, 10);

    Student s2 = get_student_1000();
    s2.dump();

    Student s3 = get_student_3000();
    s3.dump();
}