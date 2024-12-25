run: main
	./main

main: main.o
	# clang++ -v -g -Lfolly/lib -lfolly -o main main.o
	clang++ -v -g -o main main.o folly/lib/libfolly.a

main2: main2.o
	clang++ -v -g -o main2 main2.o

main.o: main.cpp
	clang++ -c -std=c++14 -g -O0 main.cpp

main2.o: main2.cc
	clang++ -c -std=c++14 -g -O0 main2.cc

xmain: main.cpp
	clang++ -Xclang -ast-print -fsyntax-only main.cpp > main.cppx
	code main.cppx

clean:
	rm -rvf *.o main
