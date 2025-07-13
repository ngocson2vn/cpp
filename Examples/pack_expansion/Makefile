# run: main
# 	./main

main: main.o
	clang++ -g -o main main.o

main.o: main.cc
	clang++ -c -std=c++17 -g -O0 main.cc

xmain: main.cc
	clang++ -std=c++17 -Xclang -ast-print -fsyntax-only main.cc > main.ccx
	code main.ccx

clean:
	rm -rvf *.o *.so main
