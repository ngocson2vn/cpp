CWD = $(shell pwd)

all:
	g++ -c -Wall -Werror -fpic task1.cpp task2.cpp
	g++ -shared -std=c++14 -o libtask1.so task1.o
	g++ -L$(CWD) -shared -std=c++14 -o libtask2.so task2.o
	g++ -L$(CWD) -Wall -o test main.cpp -ltask1 -ltask2

task3: task3.h task3.cpp
	g++ -c -Wall -Werror -fpic task3.cpp
	g++ -shared -std=c++14 -o libtask3.so task3.o
	g++ -L$(CWD) -Wall -o task3 test_task3.cpp -ltask3 -ltask2 -ltask1

clean:
	rm -fv *.o *.so