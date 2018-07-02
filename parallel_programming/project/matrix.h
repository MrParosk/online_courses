#ifndef MATRIX_H
#define MATRIX_H

typedef struct{
    float* values;
    unsigned int rows;
    unsigned int cols;
} matrix;

matrix* create_matrix(unsigned int rows, unsigned int cols);

void print_matrix(matrix* matrix_pointer);

void fill_matrix_random(matrix* matrix_pointer);

void fill_matrix_values(matrix* matrix_pointer, float* array);

void transpose_matrix(matrix* matrix_pointer);

matrix* matrix_vector_multiply(matrix* matrix_pointer, matrix* vector_pointer);

#endif
