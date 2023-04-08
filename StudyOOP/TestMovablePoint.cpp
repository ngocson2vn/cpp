// Test Driver Program for MovablePoint (TestMovablePoint.cpp)
#include <iostream>
#include <stdio.h>
#include "MovablePoint.h" // included "Point.h"
using namespace std;

int main(int argc, char** argv) {
	Point p1(4, 5); // superclass
	p1.print();     // Point @ (4, 5)
	cout << endl;

	MovablePoint mp1(11, 22); //subclass, default speed
	mp1.print(); // MovablePoint @ (11, 22) Speed (0, 0)
	cout << endl;
	mp1.setXSpeed(8);
	mp1.move();
	mp1.print(); // MovablePoint @ (11, 22) Speed (8, 0)
	cout << endl;

	MovablePoint mp2(11, 22, 33, 44);
	mp2.print(); // MovablePoint @ (11, 22) Speed (33, 44)
	cout << endl;
	mp2.move();
	mp2.print(); // MovablePoint @ (44, 66) Speed (33, 44)
	cout << endl;

	cout << endl;
	cout << "Create mp3" << endl;
	MovablePoint mp3;

	return getchar();
}
