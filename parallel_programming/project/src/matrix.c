#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "matrix.h"
#include "utils.h"

matrix* create_matrix(const unsigned int rows, const unsigned int cols){
    matrix* A = (matrix*) malloc(sizeof(matrix));
    A->rows = rows;
    A->cols = cols;

    A->values = (float*) malloc((rows * cols) * sizeof(float));

    for(unsigned int i = 0; i < rows * cols; i++){
        A->values[i] = 0.0f;
    }

    return A;
}

void free_matrix(matrix* matrix_ptr){
    matrix_ptr->rows = 0;
    matrix_ptr->cols = 0;
    free(matrix_ptr->values);
    free(matrix_ptr);
}

void equal_matrix(matrix* lhs, matrix* rhs){
    // lhs = rhs

    for(unsigned int i = 0; i < lhs->rows * lhs->cols; i++){
        lhs->values[i] = rhs->values[i];
    }
}

void print_matrix(matrix* matrix_pointer){
    for(unsigned int i = 0; i < matrix_pointer->rows; i++){
        for(unsigned int j = 0; j < matrix_pointer->cols; j++){
            printf("%0.3f ", matrix_pointer->values[j + i * matrix_pointer->cols]);
        }
        printf("\n");
    }
}

void fill_matrix_random(matrix* matrix_pointer){
    for(unsigned int i = 0; i < matrix_pointer->rows * matrix_pointer->cols; i++){
        matrix_pointer->values[i] = uniform();
    }
}

void fill_matrix_values(matrix* matrix_pointer, const float* array){
    for(unsigned int i = 0; i < matrix_pointer->rows * matrix_pointer->cols; i++){
        matrix_pointer->values[i] = array[i];
    }
}

void transpose_matrix(matrix* matrix_pointer){
    float temp_values[matrix_pointer->rows * matrix_pointer->cols];
    memcpy(temp_values, matrix_pointer->values, sizeof(float)*(matrix_pointer->rows * matrix_pointer->cols));

    for(unsigned int i = 0; i < matrix_pointer->rows; i++){
        for(unsigned int j = 0; j < matrix_pointer->cols; j++){
            matrix_pointer->values[i + j * matrix_pointer->rows] = temp_values[j + i * matrix_pointer->cols];
        }
    }

    unsigned int temp_rows = matrix_pointer->rows;
    unsigned int temp_cols = matrix_pointer->cols;

    matrix_pointer->cols = temp_rows;
    matrix_pointer->rows = temp_cols;
}
