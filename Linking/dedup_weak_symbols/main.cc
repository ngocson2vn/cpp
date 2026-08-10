#include "point.h"
#include "square.h"
#include "rectangular.h"

int main() {
  // Create a Square
  Geometry::Point s_p1(10, 10);
  Geometry::Point s_p2(20, 10);
  Geometry::Point s_p3(10, 20);
  Geometry::Point s_p4(20, 20);
  auto s = Geometry::Square(s_p1, s_p2, s_p3, s_p4);
  std::cout << "S: " << s << "\n\n";
  

  // Create a Rectangular
  Geometry::Point r_p1(10, 10);
  Geometry::Point r_p2(40, 10);
  Geometry::Point r_p3(10, 20);
  Geometry::Point r_p4(40, 20);
  auto r = Geometry::Rectangular(r_p1, r_p2, r_p3, r_p4);
  std::cout << "R: " << r << "\n\n";

  return 0;
}
