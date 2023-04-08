/*
 Test Substituting a subclass instance to a superclass reference (TestSubstitution.cpp)
 */
#include <iostream>
#include "MovablePoint.h" // included "Point.h"
using namespace std;

int TestSubstitution() {
	// Using Object Pointer
	Point* ptrP1 = new MovablePoint(11, 12, 13, 14);
	
	// Case 1:
	//     print is not a virtual function
	//     output: Point @ (11, 12) - Run superclass version!!
	// Case 2:
	//     print is a virtual function
	//     output: MovablePoint @ (11, 12) Speed (13, 14)
	ptrP1->print();
	cout << endl;

	//ptrP1->move(); // error: 'class Point' has no member named 'move'
	delete ptrP1;

	// Using Object Reference
	MovablePoint mp2(21, 22, 23, 24);
	Point& p2 = mp2; // upcast

	// Case 1:
	//     print is not a virtual function
	//     output: Point @ (21, 22) - Run superclass version!!
	// Case 2:
	//     print is a virtual function
	//     output: MovablePoint @ (21, 22) Speed (23, 24)
	p2.print();
	cout << endl;
	//p2.move(); // error: 'class Point' has no member named 'move'

	// Using object with explicit constructor
	Point p3 = MovablePoint(31, 32, 33, 34); // upcast
	p3.print(); // Point @ (31, 32) - Run superclass version!!
	cout << endl;
	//p3.move(); // error: 'class Point' has no member named 'move'

	getchar();
	return 0;
}