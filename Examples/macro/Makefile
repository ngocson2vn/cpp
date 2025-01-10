# run: main
# 	./main

main: main.o
	clang++ -g -o main main.o

main2: main2.cc
	clang++ -std=c++14 -g -O0 -o main2 main2.cc

main.o: main.cc
	clang++ -E -c -std=c++14 -g -O0 main.cc

xmain: main.cc
	clang++ -Xclang -ast-print -fsyntax-only main.cc > main.ccx
	code main.ccx

clean:
	rm -rvf *.o *.so *.cpp main main2
