output: main.o matrix.o utils.o cpu_gradient_descent.o
	gcc main.o matrix.o utils.o cpu_gradient_descent.o -o output

main.o: main.c
	gcc -c main.c

matrix.o: src/matrix.c src/matrix.h
	gcc -c src/matrix.c

cpu_gradient_descent.o: src/cpu_gradient_descent.c src/cpu_gradient_descent.h
	gcc -c src/cpu_gradient_descent.c

utils.o: src/utils.c src/utils.h
	gcc -c src/utils.c

clean:
	rm *.o output