#include <stddef.h>
#include <iostream>
using namespace std;

//#define MOVABLE

class RemoteInteger {
public:
	RemoteInteger() {
		log("Default constructor.");
		m_p = nullptr;
	}

	explicit RemoteInteger(const int n) {
		log("Unary constructor.");
		m_p = new int(n);
	}

	RemoteInteger(const RemoteInteger& other) {
		log("Copy constructor.");
		if (other.m_p) {
			m_p = new int(*other.m_p);
		} else {
			m_p = nullptr;
		}
	}

#ifdef MOVABLE
	RemoteInteger(RemoteInteger&& other) {
		log("MOVE CONSTRUCTOR.");
		m_p = other.m_p;
		other.m_p = nullptr;
	}
#endif

	RemoteInteger& operator=(const RemoteInteger& other) {
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
	RemoteInteger& operator=(RemoteInteger&& other) {
		log("MOVE ASSIGNMENT OPERATOR.");
		if (this != &other) {
			delete m_p;

			m_p = other.m_p;
			other.m_p = nullptr;
		}

		return *this;
	}
#endif

	~RemoteInteger() {
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

RemoteInteger square(const RemoteInteger& r) {
	const int i = r.get();

	return RemoteInteger(i * i);
}

RemoteInteger square2(const RemoteInteger& r) {
	const int i = r.get();
	RemoteInteger ret = RemoteInteger(i * i);
	
	return ret;
}

RemoteInteger square3() {
	RemoteInteger ret(20);

	return ret;
}

int TestRvalueMoveSemantics() {
//int main() {
	RemoteInteger a(8);
	cout << a.get() << endl << endl;

	RemoteInteger b(10);
	cout << b.get() << endl << endl;

	//square(a);

	b = square(a);
	cout << b.get() << endl << endl;

	b = square2(a);
	cout << b.get() << endl << endl;

	b = square3();
	cout << b.get() << endl << endl;

	return getchar();
}