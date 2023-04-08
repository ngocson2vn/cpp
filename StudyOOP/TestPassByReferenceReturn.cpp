/* Passing back return value using reference (TestPassByReferenceReturn.cpp) */
#include <iostream>
using namespace std;

int& squareRef(int& rNumber);

int TestPassByReferenceReturn() {
	int number1 = 8;
	cout << "In main() &number1: " << &number1 << endl;
	int& result = squareRef(number1);
	cout << "In main() &result: " << &result << endl;
	cout << result << endl;
	cout << number1 << endl;

	return getchar();
}

int& squareRef(int& rNumber) {
	cout << "In squareRef(): " << &rNumber << endl;
	rNumber *= rNumber;
	return rNumber;
}