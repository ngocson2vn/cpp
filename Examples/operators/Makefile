all: arrow
	./arrow

arrow: arrow.cpp
	clang++ -g -O0 -o arrow arrow.cpp

pointer_to_member: pointer_to_member.cc
	clang++ -g -O0 -o pointer_to_member pointer_to_member.cc

xarrow: arrow.cpp
	clang++ -Xclang -ast-print -fsyntax-only arrow.cpp > arrow.cppx
	code arrow.cppx

