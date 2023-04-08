// Header for Moving 3D Points with int coords (MovablePoint.h)
#ifndef MOVING_POINT_H
#define MOVING_POINT_H

// Include header of base class
#include "Point.h"

// MovablePoint is a subclass of Point
class MovablePoint : public Point {
private:
	int xSpeed;
	int ySpeed;

public:
	MovablePoint();
	MovablePoint(int x, int y, int xSpeed = 0, int ySpeed = 0);
	int getXSpeed() const;
	int getYSpeed() const;
	void setXSpeed(int xSpeed);
	void setYSpeed(int ySpeed);
	void move();
	void print() const;
};
#endif
