#include <iostream>
#include "Base.h"
#include "Singleton.h"

void print(const Base& objBase) {
	std::cout << "Age: " << objBase.getAge() << std::endl;
	std::cout << "Height: " << objBase.getHeight() << std::endl;
}

int TestBase(void) {
	{
		Base base;
		base.setAge(29);
		base.setHeight(170);

		std::cout << "The 1st base instance:" << std::endl;
		for (int i = 0; i < 10; i++) {
			std::cout << "\t" << base.count() << std::endl;
		}
		std::cout << "Age: " << base.getAge() << std::endl;
		std::cout << "Height: " << base.getHeight() << std::endl;
	} // base is out of scope and it's destructor is called

	std::cout << std::endl;
	Base base2;
	std::cout << "The 2nd base instance:" << std::endl;
	for (int i = 0; i < 10; i++) {
		std::cout << "\t" << base2.count() << std::endl;
	}
	std::cout << "Age: " << base2.getAge() << std::endl;
	std::cout << "Height: " << base2.getHeight() << std::endl;
	std::cout << std::endl;

	std::cout << std::endl;
	std::cout << "Creating Base object base3" << std::endl;
	Base* base3 = new Base();
	base3->setAge(29);
	delete base3;
	std::cout << std::endl;

	Singleton* s1 = Singleton::getInstance();
	s1->setAge(10);
	s1->print();

	Singleton* s2 = Singleton::getInstance();
	s2->print();

	Singleton* s3 = Singleton::getInstance();
	s3->print();

	return getchar();
}