#include <string>
#include <iostream>
using namespace std;

class Dummy {
  public:
    std::string content;

  public:
    Dummy(const std::string c) : content(c) {
      std::cout << "Dummy Constructor: " << this << std::endl;
    }

    ~Dummy() {
      std::cout << "Dummy Destructor: " << this << std::endl;
      content = "";
    }
};

class Sandbox {
  public:
    Sandbox(const Dummy& d) : member(d) {
      std::cout << "d: " << &d << std::endl;
      std::cout << "member: " << &member << std::endl;
    }
    const Dummy& member;
};

int main() {
  Sandbox sandbox(Dummy("Dummy1"));
  std::cout << "sandbox.member: " << &sandbox.member << std::endl;
  std::cout << "The answer is: " << sandbox.member.content << std::endl;

  std::cout << std::endl;

  Dummy d2("Dummy2");
  Sandbox sandbox2(d2);
  std::cout << "sandbox.member: " << &sandbox2.member << std::endl;
  std::cout << "The answer is: " << sandbox2.member.content << std::endl;
  return 0;
}
