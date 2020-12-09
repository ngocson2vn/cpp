#include <iostream>
using namespace std;

class C {
public:
	int i;

	// an explicit copy constructor
	//explicit C(const C& other) {
	C(const C& other) {
		cout << "[" << this << "] " << "in the copy constructor" << endl;
		i = other.i;
	}

	// and explicit constructor
	explicit C(int i) {
		cout << "[" << this << "] " << "in the constructor" << endl;
		this->i = i;
	}

	C() {
		i = 0;
	}
};

class C2 {
public:
	int i;
	
	// an explicit constructor
	//explicit C2(int i) {
	C2(int i) {
		this->i = i;
	}
};

C f(C c) {
	c.i = 2;
	return c; // first call to copy constructor
}

void f2(C2 c2) {
	cout << c2.i << endl;
}

void g(int i) {
	f2(i);

	// try the following line instead
	// f2(C2(i));
}

int TestExplicitKeyword() {
//int main() {
	C c, d;
	d = f(c); // c is copied

	return getchar();
}