#include "point.h"
#include <cassert>

namespace Geometry {

struct Square {
  Point p1;
  Point p2;
  Point p3;
  Point p4;

  Square(Point in_p1, Point in_p2, Point in_p3, Point in_p4)
    : p1(in_p1), p2(in_p2), p3(in_p3), p4(in_p4) {
    assert(p1.x == p3.x);
    assert(p2.x == p4.x);
    assert(p1.y == p2.y);
    assert(p3.y == p4.y);
    assert(p2.x - p1.x == p3.y - p1.y);
    assert(p2.x - p1.x == p4.y - p2.y);
  }

  void print();

  friend std::ostream& operator<<(std::ostream& os, const Square& s) {
    auto& p1 = s.p1;
    auto& p2 = s.p2;
    auto& p3 = s.p3;
    auto& p4 = s.p4;

    os << "{" << "p1" << p1 << ", p2" << p2 << ", p3" << p3 << ", p4" << p4 << "}";
    return os;
  }
};

} // namespace Geometry