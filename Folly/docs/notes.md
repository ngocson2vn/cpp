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
