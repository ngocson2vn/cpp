# run: main
# 	./main

main: logger.o main.o
	clang++ -g -o main logger.o main.o

logger.o: logger.cc
	clang++ -c -std=c++17 -g -O0 logger.cc

main.o: main.cc
	clang++ -c -std=c++17 -g -O0 main.cc

xmain: main.cc
	clang++ -std=c++17 -Xclang -ast-print -fsyntax-only main.cc > main.cxx
	code main.cxx

clean:
	rm -rvf *.o *.so main main.cxx
