# run: main
# 	./main

main: main.o
	gcc -g -o main main.o

main.o: main.c
	gcc -c -g -O0 main.c

clean:
	rm -rvf *.o *.so main
