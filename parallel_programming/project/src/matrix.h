#ifndef MATRIX_H
#define MATRIX_H

typedef struct{
    float* values;
    unsigned int rows;
    unsigned int cols;
} matrix;

matrix* create_matrix(const unsigned int rows, const unsigned int cols);

void free_matrix(matrix* matrix_ptr);

void equal_matrix(matrix* lhs, matrix* rhs);

void print_matrix(matrix* matrix_pointer);

void fill_matrix_random(matrix* matrix_pointer);

void fill_matrix_values(matrix* matrix_pointer, const float* array);

void transpose_matrix(matrix* matrix_pointer);

matrix* matrix_vector_multiply(matrix* matrix_pointer, matrix* vector_pointer);

#endif
