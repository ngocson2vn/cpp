default: case1
	./main

case1: main.cc
	clang++ -std=c++17 -g -O0 -DCASE_1 -o main main.cc

case2: main.cc
	clang++ -std=c++17 -g -O0 -DCASE_2 -o main main.cc

xmain: main.cc
	clang++ -std=c++17 -DCASE_1 -Xclang -ast-print -fsyntax-only main.cc > main.cxx
	code main.cxx

clean:
	rm -fv main main.cxx
