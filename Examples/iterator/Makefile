all: test
	./test

test: test.cpp
	clang++ -g -O0 -o test test.cpp

expand: test.cpp
	clang++ -Xclang -ast-print -fsyntax-only test.cpp > test.cppx
	code test.cppx

