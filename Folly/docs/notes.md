# Understanding usage of std::move in Facebook folly example
Question:
I am a Python programmer. I'm trying to get more into C++ and wanted to do that via understanding and using another library. I am familiar with futures/threadpools in Python, and wanted to explore the same in C++, so I chose to look at the folly library.

I've only been using this for two days, but below I create a promise, try to fulfill it via an executor which returns a future. I then want to chain that future with another action which is to call do_something . What I don't understand is why do I have to use move(f)? I have highlighted the line below:
```C++
#include <folly/futures/Future.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <iostream>
#include <chrono>
#include <thread>

using namespace folly;
using namespace std;

void do_something(Unit x) {
  std::this_thread::sleep_for(std::chrono::seconds(5));
}

int main(){
  folly::CPUThreadPoolExecutor executor(4);

  Promise<Unit> p;
  Future<Unit> f = p.getFuture().via(&executor);
  auto f_inal = move(f).thenValue(do_something);   //<<< why is move needed here?

  p.setValue();
  f_inal.wait();
  cout << "DONE" << endl;
  return 0;
}
```

This thenValue is a real doozy. Doesn't the dot in move(f).thenValue(do_something) mean "call member function of"? Here is the signature of thenValue from the source code (sorry I don't know how to make code blocks in replies)
```C++
template <typename R, typename... Args>
auto thenValue(R (&func)(Args...)) && {
  return std::move(*this).thenValue(&func);
}
```

Answer:
if you're not already familiar with C++ this kind of thing can be super confusing. I'm going to try and explain it in more detail in case it's helpful.

Member functions have a hidden argument, the object they're operating on (this). In the space after the parameters of a member function, you will often see 'const' being specified. This means if the object itself is const, the method will be chosen over a non-const method (if it exists)

In this case thenValue has a '&&' specified, which means when the object is an rvalue (such as if its a temporary or the result of a std::move) then that function will be chosen. Because thenValue knows that the object is an rvalue it can make certain decisions safely in order to optimise for things like performance.

As an example, if you had a unique_ptr member variable, it would be able to safely std::move that member variable and return it from the function. It knows the object is an rvalue, so that unique_ptr is likely about to get destroyed anyway, no harm in moving it (stealing it) before it actually is destroyed.

There are 8 ways (technical word is ref qualifiers and CV qualifiers) for member functions to be overloaded this way, these are:

&

&&

const&

const&&

volatile

volatile const

volatile const&

volatile const&&

(leaving it blank isn't valid when you do overloading, all of them then have to be explicitly defined)

Volatile isn't normally useful, so I wouldn't worry about it. & is the default when you don't specify anything. Const is the most common non-default. Ultimately a good C++ library will need to deal with these possibilities, but it's not something you'll see often in normal user code.

I'm also sure that given how complex C++ is I've gotten something wrong, hopefully someone catches it :)


# Error undefined symbol: _ZN5folly3f146detail12F14LinkCheckILNS1_17F14IntrinsicsModeE1EE5checkEv
```C++
folly::f14::detail::F14LinkCheck<(folly::f14::detail::F14IntrinsicsMode)1>::check()

// folly/include/folly/container/detail/F14Table.h
F14LinkCheck<getF14IntrinsicsMode()>::check();

// folly/include/folly/container/detail/F14IntrinsicsAvailability.h
static constexpr F14IntrinsicsMode getF14IntrinsicsMode() {
#if !FOLLY_F14_VECTOR_INTRINSICS_AVAILABLE
  return F14IntrinsicsMode::None;
#elif !FOLLY_F14_CRC_INTRINSIC_AVAILABLE
  return F14IntrinsicsMode::Simd;
#else
  return F14IntrinsicsMode::SimdAndCrc;
#endif
}

// F14 has been implemented for SSE2 and NEON (so far)
#if FOLLY_SSE >= 2 || FOLLY_NEON
#define FOLLY_F14_VECTOR_INTRINSICS_AVAILABLE 1
#else
#define FOLLY_F14_VECTOR_INTRINSICS_AVAILABLE 0
#pragma message                                                      \
    "Vector intrinsics / F14 support unavailable on this platform, " \
    "falling back to std::unordered_map / set"
#endif

#ifndef FOLLY_SSE
# if defined(__SSE4_2__)
#  define FOLLY_SSE 4
#  define FOLLY_SSE_MINOR 2
# elif defined(__SSE4_1__)
#  define FOLLY_SSE 4
#  define FOLLY_SSE_MINOR 1
# elif defined(__SSE4__)
#  define FOLLY_SSE 4
#  define FOLLY_SSE_MINOR 0
# elif defined(__SSE3__)
#  define FOLLY_SSE 3
#  define FOLLY_SSE_MINOR 0
# elif defined(__SSE2__)
#  define FOLLY_SSE 2
#  define FOLLY_SSE_MINOR 0
# elif defined(__SSE__)
#  define FOLLY_SSE 1
#  define FOLLY_SSE_MINOR 0
# else
#  define FOLLY_SSE 0
#  define FOLLY_SSE_MINOR 0
# endif
#endif
```

## Fix 
Step 1: Check available API:
```Bash
mkdir tmp
cd tmp
cp /path/to/libfolly.a .
ar x libfolly.a
readelf -sW F14Table.cpp.o | c++filt | grep F14LinkCheck
     6: 0000000000000000     1 FUNC    GLOBAL DEFAULT    2 folly::f14::detail::F14LinkCheck<(folly::f14::detail::F14IntrinsicsMode)2>::check()
# In this case F14IntrinsicsMode = 2
```

Step 2: Define required variables at compile time
```C++
# folly/include/folly/Portability.h
#ifndef __SSE4_2__
#define __SSE4_2__
#endif
```
