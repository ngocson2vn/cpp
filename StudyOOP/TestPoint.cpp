// A test driver program for Point class (TestPoint.cpp)
#include <iostream>
#include "Point.h"
using namespace std;

void TestPoint() {
	//Point p1 = Point(10, 10);
	//p1.setX(20);
	//p1.setY(40);
	//p1.print();

	//Point* p1 = new Point(1, 2);
	//p1->print();
	//delete p1;

	// Invoke default constructor
	Point p1; // OR Point p1 = Point(); NOT Point p1();

	// Invoke constructor
	Point p2(2, 2); // OR Point p2 = Point(2, 2);
	p1.print(); // Point @ (0, 0)
	cout << endl;
	p2.print(); // Point @ (2, 2)
	cout << endl;

	// Object Pointers with dynamic allocation
	// Declare tow Point pointers
	Point* ptrP3;
	Point* ptrP4;

	ptrP3 = new Point(); // Dynamically allocate storage via new
	                     // with default constructor

	ptrP4 = new Point(4, 4);
	ptrP3->print(); // Point @ (0, 0)
	cout << endl;

	ptrP4->print(); // Point @ (4, 4)
	cout << endl;

	delete ptrP3; // Remove storage via delete
	delete ptrP4;

	// Object Reference (Alias)
	Point& p5 = p2; // Reference (alias) to an existing object
	p5.print(); // Point @ (2, 2)
	cout << endl;

	/***************************
	 * ARRAYS                  *
	 ***************************/

	// Array of Objects - Static Memory Allocation
	// Array of Point objects
	// Use default constructor for all elements of the array
	Point ptsArray1[2];
	ptsArray1[0].print(); // Point @ (0, 0)
	cout << endl;
	ptsArray1[1].setXY(11, 11);
	(ptsArray1 + 1)->print(); // Point @ (11, 11)
	                          // same as ptsArray1[1].print()
	cout << endl;

	// Declare another array of Point objects
	// Initialize array elements via constructor
	Point ptsArray2[3] = { Point(21, 21), Point(22, 22), Point() };
	ptsArray2->print(); // Point @ (21, 21)
	cout << endl;
	(ptsArray2 + 2)->print(); // Point @ (0, 0)
	cout << endl;

	// Array of Object Pointers - Need to allocate elements dynamically
	Point* ptrPtsArray3 = new Point[2];
	ptrPtsArray3[0].setXY(31, 31);
	ptrPtsArray3->print(); // Point @ (31, 31)
	                       // same as ptrPtsArray3[0].print()
	cout << endl;

	(ptrPtsArray3 + 1)->setXY(32, 32);
	ptrPtsArray3[1].print(); // Point @ (32, 32)
	cout << endl;

	delete[] ptrPtsArray3; // Free storage

	getchar();
}