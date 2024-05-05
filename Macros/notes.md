# Test Macro Expansion
```bash
cpp macro1.cpp > macro1.i
```

# Token Concatenation with '##'
```c++
#define INTERNAL_REGISTER_LOCAL_DEVICE_FACTORY_NAME(ctr) ___##ctr##__object_

INTERNAL_REGISTER_LOCAL_DEVICE_FACTORY_NAME(0);
// will be expanded to ___0__object_
```
