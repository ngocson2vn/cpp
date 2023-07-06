#include "Base.h"

class Derived: public Base {
  public:
    virtual void someMethod() override;
    virtual void someOtherMethod();
};