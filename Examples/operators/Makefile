all: arrow
	./arrow

arrow: arrow.cpp
	clang++ -g -O0 -o arrow arrow.cpp

xarrow: arrow.cpp
	clang++ -Xclang -ast-print -fsyntax-only arrow.cpp > arrow.cppx
	code arrow.cppx

