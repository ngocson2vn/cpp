all: main
	./main

main: main.cpp
	clang++ -g -O0 -o main main.cpp

xmain: main.cpp
	clang++ -Xclang -ast-print -fsyntax-only main.cpp > main.cppx
	code main.cppx

