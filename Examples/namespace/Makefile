run: main
	./main

main: main.o
	clang++ -g -o main main.o && echo

main.o: main.cpp
	clang++ -c -std=c++17 -g -O0 main.cpp

xmain: main.cpp
	clang++ -Xclang -ast-print -fsyntax-only main.cpp > main.cppx
	code main.cppx

clean:
	rm -rvf *.o *.so main
