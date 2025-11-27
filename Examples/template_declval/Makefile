default: clean main

main: main.cpp
	clang++ -std=c++17 -g -O0 -o main main.cpp

xmain: main.cpp
	clang++ -Xclang -ast-print -fsyntax-only main.cpp > main.cppx
	code main.cppx

clean:
	rm -rf *.o *.cppx main