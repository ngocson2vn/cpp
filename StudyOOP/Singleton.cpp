#include <iostream>
#include "Singleton.h"

Singleton* Singleton::instance = nullptr;

Singleton::Singleton() { }

Singleton* Singleton::getInstance() {
	
	//static Singleton* instance = nullptr;

	if (instance != nullptr) {
		std::cout << "The previous instance" << std::endl;
		return instance;
	} else {
		instance = new Singleton();
		std::cout << "The first instance" << std::endl;
		return instance;
	}
}

void Singleton::setAge(int age) {
	this->age = age;
}

void Singleton::print() const {
	std::cout << "Age = " << this->age << std::endl;
}

Singleton::~Singleton()
{
}
