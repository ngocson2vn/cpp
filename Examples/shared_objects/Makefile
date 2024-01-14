CWD = $(shell pwd)

all:
	g++ -c -Wall -Werror -fpic task1.cpp task2.cpp
	g++ -shared -std=c++14 -o libtask1.so task1.o
	g++ -L$(CWD) -shared -std=c++14 -o libtask2.so task2.o -ltask1
	g++ -L$(CWD) -Wall -o test main.cpp -ltask2
clean:
	rm -f *.o *.so