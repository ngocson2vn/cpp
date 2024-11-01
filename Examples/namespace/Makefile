run: main
	./main

main: main.o
	clang++ -v -g -o main main.o

main.o: main.cpp
	clang++ -c -std=c++14 -g -O0 main.cpp

xmain: main.cpp
	clang++ -Xclang -ast-print -fsyntax-only main.cpp > main.cppx
	code main.cppx

clean:
	rm -rvf *.o *.so main
