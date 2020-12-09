#include <stddef.h>
#include <iostream>
using namespace std;

#define MOVABLE

class RemoteInteger2 {
public:
	RemoteInteger2() {
		log("Default constructor.");
		m_p = nullptr;
	}

	explicit RemoteInteger2(const int n) {
		log("Unary constructor.");
		m_p = new int(n);
	}

	RemoteInteger2(const RemoteInteger2& other) {
		log("Copy constructor.");
		m_p = nullptr;
		*this = other; // call copy assignment operator
	}

#ifdef MOVABLE
	RemoteInteger2(RemoteInteger2&& other) {
		log("MOVE CONSTRUCTOR.");
		m_p = nullptr;
		*this = other; // call move assignment operator but this is WRONG
	}
#endif

	RemoteInteger2& operator=(const RemoteInteger2& other) {
		log("Copy assignment operator.");
		if (this != &other) {
			delete m_p;

			if (other.m_p) {
				m_p = new int(*other.m_p);
			} else {
				m_p = nullptr;
			}
		}

		return *this;
	}

#ifdef MOVABLE
	RemoteInteger2& operator=(RemoteInteger2&& other) {
		log("MOVE ASSIGNMENT OPERATOR.");
		if (this != &other) {
			delete m_p;

			m_p = other.m_p;
			other.m_p = nullptr;
		}

		return *this;
	}
#endif

	~RemoteInteger2() {
		log("Destructor.");
		delete m_p;
		m_p = nullptr;
	}

	int get() const {
		return m_p ? *m_p : 0;
	}

private:
	int* m_p;

	void log(const char* msg) {
		cout << "[" << this << "] " << msg << endl;
	}
};

RemoteInteger2 frumple(const int n) {
	if (n == 1729) {
		return RemoteInteger2(1729);
	}

	RemoteInteger2 ret(n * n);
	
	return ret;
}

int TestRvalueWrongMove() {
//int main() {

	//RemoteInteger2 a(10);
	//RemoteInteger2 b = a;

	RemoteInteger2 x = frumple(5);
	cout << x.get() << endl << endl;

	RemoteInteger2 y(1729);
	cout << y.get() << endl << endl;

	return getchar();
}