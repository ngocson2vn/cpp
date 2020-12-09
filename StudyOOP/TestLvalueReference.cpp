#include <iostream>
using namespace std;

void square(int& n) {
	n = n * n;
}

//void square(int* p) {
//	*p = (*p) * (*p);
//}

int TestLvalueReference() {
//int main() {
	int a = 10;
	int b = 20;

	square(a);
	cout << "a: " << a << endl;

	//square(&b);
	//cout << "b: " << b << endl;

	square(10);

	return getchar();
}