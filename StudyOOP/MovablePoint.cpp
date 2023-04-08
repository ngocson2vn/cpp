// Implementation for Moving 3D Points with int coords (MovablePoint.cpp)
#include <iostream>

// Include header containing the class declaration
#include "MovablePoint.h"
using namespace std;

MovablePoint::MovablePoint() {
	cout << "MovablePoint constructor no parameter" << endl;
	xSpeed = 0;
	ySpeed = 0;
}

MovablePoint::MovablePoint(int x, int y, int xSpeed, int ySpeed) : Point(x, y), xSpeed(xSpeed), ySpeed(ySpeed) { }

// Getters
int MovablePoint::getXSpeed() const { return this->xSpeed; }
int MovablePoint::getYSpeed() const { return this->ySpeed; }

// Setters
void MovablePoint::setXSpeed(int xSpeed) { this->xSpeed = xSpeed; }
void MovablePoint::setYSpeed(int ySpeed) { this->ySpeed = ySpeed; }

// Functions
void MovablePoint::print() const {
	cout << "Movable";

	// Invoke base class function via scope resolution operator
	Point::print();
	
	cout << " Speed " << "(" << xSpeed << ", " << ySpeed << ")";
}

void MovablePoint::move() {
	// Subclass cannot access private member of the superclass directly
	// Need to go through the public interface
	//Point::setX(Point::getX() + this->xSpeed);
	//Point::setY(Point::getY() + this->ySpeed);

	// We can also make change to x and y as follows
	this->setX(this->getX() + this->xSpeed);
	this->setY(this->getY() + this->ySpeed);
}
