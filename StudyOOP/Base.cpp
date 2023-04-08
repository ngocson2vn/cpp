#include "Base.h"

int Base::count() {
	static int instanceCount = 0;

	return ++instanceCount;
}

//Base::Base() {
//	std::cout << "Base::Base() constructor" << std::endl;
//	this->age = 0;
//	this->height = 0;
//}

Base::Base(int age, int height)
{
	std::cout << "Base constructor:" << std::endl;
	std::cout << "\tage = " << age << ", height = " << height << std::endl;
	this->age = age;
	this->height = height;
}

int Base::getAge() const {
	return this->age;
}

void Base::setAge(int age) {
	this->age = age;
}

int Base::getHeight() const {
	return this->height;
}

void Base::setHeight(int height) {
	this->height = height;
}

Base::~Base()
{
	std::cout << "Base object is being destroyed:" << std::endl;
	std::cout << "\tage = " << age << ", height = " << height << std::endl;
}
