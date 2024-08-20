all: move
	./move

move: move.cpp
	clang++ -g -O0 -o move move.cpp

xmove: move.cpp
	clang++ -Xclang -ast-print -fsyntax-only move.cpp > move.cppx
	code move.cppx

