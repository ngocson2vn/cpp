#include <iostream>
#include <vector>

struct ClassA {
    int size;
    std::vector<std::string*> items;
};

struct ClassB {
    ClassA a;

    ClassB() = default;

    // Move Constructor
    ClassB(ClassB&& rhs) {
        std::cout << "ClassB(ClassB&& rhs):" << std::endl;
        std::cout << "  -> this: " << this << ", rhs: " << &rhs << std::endl;
        a = std::move(rhs.a);
    }

    ~ClassB() {
        std::cout << "~ClassB(): " << std::endl;
        std::cout << "  -> this: " << this << std::endl;
    }

    void init() {
        a.size = 10;
        a.items.reserve(a.size);
        for (int i = 0; i < a.size; i++) {
            std::string* item = new std::string("item");
            item->append(std::to_string(i));
            a.items.push_back(item);
        }
    }

    void dance() {
        for (auto item : a.items) {
            std::cout << *item << std::endl;
        }
    }

    bool empty() {
        std::cout << "a.size: " << a.size << ", a.items.size(): " << a.items.size() << std::endl;
        return ((a.size == 0) && (a.items.size() == 0));
    }
};

void play_rvalue(ClassB&& b) {
    b.dance();
}

void play_lvalue(ClassB b) {
    b.dance();
}

int main() {
    ClassB b;
    b.init();

    std::cout << std::endl;
    std::cout << "b: " << &b << std::endl;
    play_rvalue(std::move(b));
    // play_lvalue(std::move(b));

    std::cout << std::endl;
    std::string empty = b.empty() ? "true" : "false";
    std::cout << "b is empty: " << empty << std::endl;
    std::cout << std::endl;

    std::cout << "===============================================" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << std::endl;
        std::cout << "i = " << i << std::endl;
        ClassB b;
        b.init();
        std::cout << "Done using b" << std::endl;
    }
}
