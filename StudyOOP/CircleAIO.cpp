#include <iostream>
#include <string>
using namespace std;

class Circle {
private:
	double radius;
	string color;

public:
	Circle(double r = 1.0, string c = "red") {
		radius = r;
		color = c;
	}

	double getRadius() const {
		return radius;
	}

	void setRadius(double r) {
		radius = r;
	}

	string getColor() const {
		return color;
	}

	double getArea() const {
		return radius * radius * 3.1416;
	}

	void display() const {
		cout << "Radius = " << getRadius()
			 << " Area = " << getArea()
			 << " Color = " << getColor() << endl;
	}
};

int main() {
	cout << endl;
	cout << "Construct object c1" << endl;
	Circle c1(1.2, "blue");
	cout << "&c1 = " << &c1 << endl;
	c1.display();
	cout << endl;

	cout << "Construct a new object c2 by copying object c1" << endl;
	Circle c2(c1);
	cout << "&c2 = " << &c2 << endl;
	c2.display();
	cout << endl;

	cout << ">>> Make change to c2" << endl;
	c2.setRadius(2.0);
	cout << "c1:" << endl;
	c1.display();
	cout << "c2:" << endl;
	c2.display();
	cout << endl;

	cout << "Assign object c2 to object c3" << endl;
	Circle c3 = c2;
	cout << "&c3 = " << &c3 << endl;
	c3.display();
	cout << endl;

	cout << ">>> Make change to c3" << endl;
	c3.setRadius(3.0);
	cout << "c2:" << endl;
	c2.display();
	cout << "c3:" << endl;
	c3.display();
	
	getchar();
}