sony.o: sony.h sony.cpp
	clang++ -c -g -O0 -Wall -Werror -fpic sony.cpp

sony: sony.o
	clang++ -shared -std=c++14 -o libsony.so sony.o

run: main
	./main

main: main.o
	clang++ -v -g -o main main.o libsony.so -ldl

main.o: main.cpp
	clang++ -c -std=c++14 -g -O0 main.cpp

xmain: main.cpp
	clang++ -Xclang -ast-print -fsyntax-only main.cpp > main.cppx
	code main.cppx

clean:
	rm -rvf *.o *.so main
