// The Point class Header file (Point.h)
#ifndef POINT_H
#define POINT_H

class Point {
private:
	// Private data members
	int x, y;

public:
	// Constructor with default argument
	Point(int x, int y);

	// Getter
	int getX() const;

	// Setter
	void setX(int x);
	
	int getY() const;
	void setY(int y);
	void setXY(int x, int y);
	//void print() const;
	virtual void print() const;
};

#endif
