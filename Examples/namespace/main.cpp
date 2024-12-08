#include "detail.h"
#include "phutho.h"

using namespace sony;

int main(int argc, char** argv) {
  detail::Foo foo;
  foo.compute();

  // phutho::CamKhe ck;
  vn::phutho::CamKhe ck;
  ck.print();
}
