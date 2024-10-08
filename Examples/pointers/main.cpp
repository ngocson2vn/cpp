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

int main(int argc, char** argv) {
    const Student s1{100, "Son"};
    set_score(&s1, 10);
}