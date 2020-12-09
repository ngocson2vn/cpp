#include <stddef.h>
#include <iostream>
#include <utility>
using namespace std;

#define MOVABLE

template <typename T> struct RemoveReference {
	typedef T type;
};

template <typename T> struct RemoveReference<T&> {
	typedef T type;
};

template <typename T> struct RemoveReference<T&&> {
	typedef T type;
};

template <typename T> typename RemoveReference<T>::type&& Move(T&& t) {
	return (typename RemoveReference<T>::type&&)t;
}

class RemoteIntegerMovable {
public:
	RemoteIntegerMovable() {
		log("Default constructor.");
		m_p = nullptr;
	}

	explicit RemoteIntegerMovable(const int n) {
		log("Unary constructor.");
		m_p = new int(n);
	}

	RemoteIntegerMovable(const RemoteIntegerMovable& other) {
		log("Copy constructor.");
		m_p = nullptr;
		*this = other; // call copy assignment operator
	}

#ifdef MOVABLE
	RemoteIntegerMovable(RemoteIntegerMovable&& other) {
		log("MOVE CONSTRUCTOR.");
		m_p = nullptr;
		*this = Move(other); // RIGHT
	}
#endif

	RemoteIntegerMovable& operator=(const RemoteIntegerMovable& other) {
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
	RemoteIntegerMovable& operator=(RemoteIntegerMovable&& other) {
		log("MOVE ASSIGNMENT OPERATOR.");
		if (this != &other) {
			delete m_p;

			m_p = other.m_p;
			other.m_p = nullptr;
		}

		return *this;
	}
#endif

	~RemoteIntegerMovable() {
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

RemoteIntegerMovable frumple(const int n) {
	if (n == 1729) {
		return RemoteIntegerMovable(1729);
	}

	RemoteIntegerMovable ret(n * n);
	
	return ret;
}

//int TestRvalueRightMove() {
int main() {

	//RemoteIntegerMovable a(10);
	//RemoteIntegerMovable b = a;

	RemoteIntegerMovable x = frumple(5);
	cout << x.get() << endl << endl;

	RemoteIntegerMovable y(1729);
	cout << y.get() << endl << endl;

	return getchar();
}