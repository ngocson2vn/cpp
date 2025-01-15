# run: main
# 	./main

main: main.o
	clang++ -g -o main main.o

main.o: main.cc
	clang++ -c -std=c++14 -g -O0 main.cc

main2: main2.cc
	clang++ -std=c++17 -g -o main2 main2.cc

xmain: main.cc
	clang++ -Xclang -ast-print -fsyntax-only main.cc > main.ccx
	code main.ccx

clean:
	rm -rvf *.o *.so main main2
