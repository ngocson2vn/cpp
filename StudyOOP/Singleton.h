#pragma once
class Singleton
{
private:
	Singleton();
	static Singleton* instance;
	int age;
public:
	static Singleton* getInstance();
	void setAge(int age);
	void print() const;
	~Singleton();
};

