# run: main
# 	./main

main: main.cc
	clang++ -std=c++17 -g -o main main.cc

xmain: main.cc
	clang++ -Xclang -ast-print -fsyntax-only main.cc > main.ccx
	code main.ccx

clean:
	rm -rvf *.o *.so main
