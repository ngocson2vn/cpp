# Macro
```C++
#define f(a,b) a##b
#define g(a) #a
#define h(a) g(a)
```
An argument is macro-replaced before it is substituted into the replacement list, except where it appears as the operand of # (stringize) or ## (concatenate).

`h(f(1,2))`:  
In your macro h, the parameter a is not an argument of one of those two operators, so the argument is macro-replaced and then substitued into the replacement list. That is, the argument f(1,2) is macro replaced to become 1##2, and then to 12, and then it is substituted into g(12), which is (again) macro-replaced to become "12".

`g(f(1,2))`: 
When you invoke g directly, the parameter a is an argument of the # operator, so its argument is not macro-replaced before subsitution: f(1,2) is substituted directly into the replacement list, yielding "f(1,2)".

Another example:
```C++
#define CUDA_ARCH_STR(x) "__CUDA_ARCH__: " #x
#define STRINGIZE_CUDA_ARCH(x) CUDA_ARCH_STR(x)
#pragma message(STRINGIZE_CUDA_ARCH(__CUDA_ARCH__) ", __CUDA_ARCH_FEAT_SM90_ALL is defined")
```
Step 1: The macro `__CUDA_ARCH__` will be expanded to `900`, then be substituted into the body of the macro `STRINGIZE_CUDA_ARCH(x)`.  
`STRINGIZE_CUDA_ARCH(__CUDA_ARCH__) -> CUDA_ARCH_STR(900)`  

Step 2: The macro `CUDA_ARCH_STR(900)` will be expanded to `"__CUDA_ARCH__: " "90"`  

Finally, we got:  
```C++
#pragma message("__CUDA_ARCH__: " "90" ", __CUDA_ARCH_FEAT_SM90_ALL is defined")
```
<br/>

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
