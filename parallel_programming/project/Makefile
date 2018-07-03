output: main.o matrix.o utils.o
	gcc main.o matrix.o utils.o -o output

main.o: main.c
	gcc -c main.c

matrix.o: matrix.c matrix.h
	gcc -c matrix.c

utils.o: utils.c utils.h
	gcc -c utils.c

clean:
	rm *.o output