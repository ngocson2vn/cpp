all: main
	./main

main: main.cc
	clang++ -g -O0 -o main main.cc

xmain: main.cc
	clang++ -Xclang -ast-print -fsyntax-only main.cc > main.ccx
	code main.ccx
