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