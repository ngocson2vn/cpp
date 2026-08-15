all: main
	./main

main: main.cpp
	clang++ -O3 -o main main.cpp

xmain: main.cpp
	clang++ -Xclang -ast-print -fsyntax-only main.cpp > main.cppx
	code main.cppx

clean:
	rm -fv *.o main
