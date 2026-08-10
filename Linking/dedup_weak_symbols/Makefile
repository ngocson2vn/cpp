# run: main
# 	./main

main: main.cc square.h square.cc rectangular.h rectangular.cc
	clang++ -std=c++14 -g -O0 -o main main.cc square.cc rectangular.cc

xmain: main.cc square.h square.cc rectangular.h rectangular.cc
	clang++ -Xclang -ast-print -fsyntax-only main.cc square.cc rectangular.cc > main.cxx
	code main.cxx

clean:
	rm -rvf *.o *.so main main.cxx
