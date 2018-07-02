output: main.o matrix.o
	gcc main.o matrix.o -o output

main.o: main.c
	gcc -c main.c

matrix.o: matrix.c matrix.h
	gcc -c matrix.c

clean:
	rm *.o output