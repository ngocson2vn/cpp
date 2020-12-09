#include <iostream>
#include <string>
using namespace std;

template <typename T> struct Name;

template <> struct Name < string > {
	static const char* get() {
		return "string";
	}
};

template <> struct Name < const string > {
	static const char* get() {
		return "const string";
	}
};

template <> struct Name < string& > {
	static const char* get() {
		return "string&";
	}
};

template <> struct Name < const string& > {
	static const char* get() {
		return "const string&";
	}
};

template <> struct Name < string&& > {
	static const char* get() {
		return "string&&";
	}
};

template <> struct Name < const string&& > {
	static const char* get() {
		return "const string&&";
	}
};

template <typename T> void quark(T&& t) {
	cout << "t: " << t << endl;
	cout << "T: " << Name<T>::get() << endl;
	cout << "T&&: " << Name<T&&>::get() << endl;
	cout << endl;
}

template <typename T> void doSquare(T&& n) {
	n = n*n;
	g_val = n;
}

string strange() {
	return "strange()";
}

const string charm() {
	return "charm()";
}

int g_val = 0;

int TestRvalueReferenceCollapsing() {
//int main() {
	string up("up");
	const string down("down");

	quark(up);
	quark(down);
	quark(strange());
	quark(charm());

	int x = 10;
	doSquare(x);
	cout << x << endl;
	cout << g_val << endl;

	doSquare(10);
	cout << g_val << endl;

	return getchar();
}