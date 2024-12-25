# run: main
# 	./main

main: main.o
	g++ -g -o main main.o

main.o: main.cc
	g++ -std=c++17 -g -O0 -c main.cc

xmain: main.cc
	clang++ -Xclang -ast-print -fsyntax-only main.cc > main.cxx
	code main.ccx

clean:
	rm -rvf *.o *.so main
