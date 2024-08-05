x1: test1.cpp
	clang++ -Xclang -ast-print -fsyntax-only test1.cpp > test1.cppx

x2: test2.cpp
	clang++ -Xclang -ast-print -fsyntax-only test2.cpp > test2.cppx

x3: test3.cpp
	clang++ -Xclang -ast-print -fsyntax-only test3.cpp > test3.cppx

x4: test4.cpp
	clang++ -Xclang -ast-print -fsyntax-only test4.cpp > test4.cppx

x5: test5.cpp
	clang++ -Xclang -ast-print -fsyntax-only test5.cpp > test5.cppx

t1: test1.cpp
	clang++ -o test1 test1.cpp

t2: test2.cpp
	clang++ -o test2 test2.cpp

k1: template_keyword1.cpp
	clang++ -o k1 template_keyword1.cpp

k2: template_keyword2.cpp
	clang++ -o k2 template_keyword2.cpp

kx2: template_keyword2.cpp
	clang++ -Xclang -ast-print -fsyntax-only template_keyword2.cpp > template_keyword2.cppx

k3: typename_keyword1.cpp
	clang++ -o k3 typename_keyword1.cpp

kx3: typename_keyword2.cpp
	clang++ -Xclang -ast-print -fsyntax-only typename_keyword2.cpp > typename_keyword2.cppx
