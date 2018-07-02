#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "matrix.h"

matrix* create_matrix(unsigned int rows, unsigned int cols){
    matrix* A = (matrix*) malloc(sizeof(matrix));
    A->rows = rows;
    A->cols = cols;

    A->values = (float*) malloc((rows * cols) * sizeof(float));
    return A;
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
    for(unsigned int i=0; i < matrix_pointer->rows * matrix_pointer->cols; i++){
        // Fix with unit normal
        matrix_pointer->values[i] = 0.0f;
    }
}

void fill_matrix_values(matrix* matrix_pointer, float* array){
    // Check so that the sizes are equal

    for(unsigned int i=0; i < matrix_pointer->rows * matrix_pointer->cols; i++){
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
