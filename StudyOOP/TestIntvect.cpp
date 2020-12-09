#include <iostream>
#include "Intvec.h"

int TestIntvect() {
//int main(int argc, char* argv[]) {
	std::cout << "creating two Intvect objects...\n";
	Intvect v1(20);
	Intvect v2;
	std::cout << "ended creating two Intvect objects...\n\n";

	std::cout << "assigning lvalue...\n";
	v2 = v1;
	std::cout << "ended assigning lvalue...\n\n";

	std::cout << "assigning rvalue...\n";
	v2 = Intvect(33);
	std::cout << "ended assigning rvalue...\n\n";

	return getchar();
}