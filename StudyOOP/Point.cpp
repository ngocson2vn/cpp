// The Point class Implementation file (Point.cpp)
#include "Point.h"
#include <iostream>
using namespace std;

// Constructor - The default values are specified in the declaration
Point::Point(int x, int y) : x(x), y(y) { 
	std::cout << "Point constructor x = " << x << ", y = " << y << std::endl;
}

// Getters
int Point::getX() const { return x; }
int Point::getY() const { return y; }

// Setters
void Point::setX(int x) { this->x = x; }
void Point::setY(int y) { this->y = y; }

// Public Functions
void Point::setXY(int x, int y) { this->x = x; this->y = y; }

void Point::print() const {
	cout << "Point @ (" << x << ", " << y << ")";
}
