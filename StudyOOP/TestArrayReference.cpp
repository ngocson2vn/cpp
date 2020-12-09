#include <iostream>

void printData(int* dataArray) {

	// You cannot get correct size of array
	size_t size = sizeof(dataArray) / sizeof(*dataArray);
	std::cout << "Data: ";

	for (size_t i = 0; i < size; i++) {
		std::cout << *(dataArray + i) << " ";
	}
}

void printDataRef(const int (&dataRefA)[10]) {

	// You get correct size of array
	size_t size = sizeof(dataRefA) / sizeof(*dataRefA);
	std::cout << "Data: ";

	for (size_t i = 0; i < size; i++) {
		std::cout << *(dataRefA + i) << " ";
	}
}

int TestArrayReference() {
//int main(int argc, char* argv[]) {

	int da[10];

	for (int i = 0; i < 10; i++) {
		da[i] = i;
	}

	printData(da);
	std::cout << std::endl << std::endl;

	printDataRef(da);

	return getchar();
}