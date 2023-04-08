#ifndef BASE_H
#define BASE_H

#include <iostream>

class Base
{
private:
	int age;
	int height;
public:
	//Base();
	Base(int age = 0, int height = 0);
	static int count();
	int getAge() const;
	void setAge(int age);
	int getHeight() const;
	void setHeight(int height);
	~Base();
};

#endif