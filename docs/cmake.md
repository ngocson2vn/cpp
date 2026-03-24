# Professional CMake
Arguments are separated from each other by spaces and may be split across multiple lines:
```C++
add_executable(MyExe
  main.cpp
  src1.cpp
  src2.cpp
)
```

# ExternalProject
```Bash
include(ExternalProject)

# Define the LLVM external project
ExternalProject_Add(
  llvm-project
  GIT_REPOSITORY https://github.com/llvm/llvm-project.git
  GIT_TAG 570885128351868c1308bb22e8ca351d318bc4a1
  GIT_PROGRESS TRUE
  PREFIX ${CMAKE_BINARY_DIR}/llvm-project
  BINARY_DIR ${CMAKE_BINARY_DIR}/llvm-project/build
  LIST_SEPARATOR ","
  CONFIGURE_COMMAND 
    ${CMAKE_COMMAND} 
    "-G Ninja" 
    "../src/llvm-project/llvm"
    "-DCMAKE_CXX_FLAGS=-g"
    "-DCMAKE_CXX_FLAGS=-O0"
    "-DLLVM_ENABLE_PROJECTS=mlir,compiler-rt"
    "-DLLVM_BUILD_EXAMPLES=ON"
    "-DLLVM_TARGETS_TO_BUILD=Native,X86,NVPTX"
    "-DCMAKE_BUILD_TYPE=Debug"
    "-DLLVM_ENABLE_ASSERTIONS=ON"
    "-DCMAKE_C_COMPILER=clang"
    "-DCMAKE_CXX_COMPILER=clang++"
    "-DLLVM_ENABLE_LLD=ON"
    "-DLLVM_CCACHE_BUILD=ON"
    "-DCOMPILER_RT_BUILD_GWP_ASAN=OFF"
    "-DLLVM_INCLUDE_TESTS=OFF"
    "-DCOMPILER_RT_BUILD_SANITIZERS=ON"
  BUILD_COMMAND ${CMAKE_COMMAND} --build . -- -v
  BUILD_ALWAYS FALSE
  USES_TERMINAL_CONFIGURE TRUE
  USES_TERMINAL_DOWNLOAD TRUE
  USES_TERMINAL_BUILD TRUE
  INSTALL_COMMAND ""
)
```

# Debug mode
```Bash
-DCMAKE_BUILD_TYPE=Debug
```

# Common Issues
## Backtrace shows unknown function
```C++
frame #8: <unknown function> + 0x75f6d (0x7f889c209f6d in /path/to/libxxx.so)
```
Root cause: Because of the `-fvisibility=hidden` option. Commenting out the following CMake variables should resolve this issue.
```Bash
# set(CMAKE_C_VISIBILITY_PRESET hidden)
# set(CMAKE_CXX_VISIBILITY_PRESET hidden)
# set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)
```
By default, when you compile a shared library (like your `.so` file) on Linux or macOS, every single non-static function, class, and global variable is exposed to the outside world. The `-fvisibility=hidden` flag changes this behavior so that **everything is kept internal to the library by default**, unless you explicitly tell the compiler to expose it.

Here is a breakdown of how it works and why it is so commonly used, especially in complex Python C++ extensions.

### How It Works
When a shared library is loaded, the operating system looks at its **Dynamic Symbol Table** (`.dynsym`) to see what functions can be called by outside programs (like Python).

* **Without the flag:** The compiler puts the name and address of *every* C++ function you wrote into the dynamic symbol table.
* **With `-fvisibility=hidden`:** The compiler hides all those names. The functions still exist in the compiled machine code, and functions *inside* the library can still call each other. However, outside programs cannot see or call them, and runtime backtrace tools won't find their names.

### Why It Is Used (The Benefits)
Using this flag is considered a best practice for modern C++ shared libraries for three main reasons:

* **Prevents Name Collisions:** This is the biggest reason Python extensions use it. If your PyTorch extension uses an internal function named `calculate_matrix()`, and another completely unrelated Python package also has a C++ function named `calculate_matrix()`, loading both packages could crash Python due to a symbol clash. Hiding the symbols prevents this.
* **Reduces Binary Size and Load Time:** The dynamic symbol table can get massive, especially with C++ templates, which generate very long, complex mangled names. Hiding symbols shrinks the `.so` file significantly and makes the OS load it into memory much faster.
* **Improves Performance:** When symbols are visible globally, the compiler has to route internal function calls through a lookup table called the Procedure Linkage Table (PLT), just in case that function gets overridden by another library. When a symbol is hidden, the compiler knows it can't be overridden and optimizes the call into a direct, faster jump.

### How to Expose What You Actually Need
If everything is hidden, Python wouldn't be able to load your library at all. To fix this, developers explicitly "opt-in" the specific functions they want to share (like the initialization function Python looks for) using a special attribute in the C++ code:

```cpp
__attribute__((visibility("default"))) void my_python_init_function() {
    // This function will be visible in the dynamic symbol table
}

```
