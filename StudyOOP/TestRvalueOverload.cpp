#include <iostream>
#include <ostream>
#include <string>
using namespace std;

void meow(string& s) {
	cout << "meow(string&): " << s << endl;
}

void meow(const string& s) {
	cout << "meow(const string&): " << s << endl;
}

void meow(string&& s) {
	cout << "meow(string&&): " << s << endl;
}

void meow(const string&& s) {
	cout << "meow(const string&&): " << s << endl;
}

void purr(const string& s) {
	cout << "purr(const string&): " << s << endl;
}

void purr(string&& s) {
	cout << "purr(string&&): " << s << endl;
}

string strange() {
	return "strange()";
}

const string charm() {
	return "charm()";
}

int TestRvalueOverload() {
//int main() {
	string up("up");
	const string down("down");

	purr(up);
	purr(down);
	purr(strange());
	purr(charm());

	return getchar();
}