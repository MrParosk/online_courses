/* 
Implementing Sparse Matrix Vector multiplication (SpMV) in CUDA with one thread per row.
The matrix uses a Compressed Sparse Row (CSR) representation.
*/

#include <stdio.h>
#define MATRIX_SIZE 4
#define NON_ZERO_ELEMENTS 6
#define NUM_BLOCKS 2
#define NUM_THREADS 2

void print_vector(int* array){
    for(unsigned int i = 0; i < MATRIX_SIZE; i++){
            printf("%d ", array[i]);
    }
}

__global__ void spmv(const int* value, const int* index, const int* row_ptr, int* d_in, int* d_out){
    const unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;

    if(row < MATRIX_SIZE){
        int start_row = row_ptr[row];
        int end_row;

        // Fixing special case for the last row
        if(row == MATRIX_SIZE - 1){
            end_row = NON_ZERO_ELEMENTS;
        }else{
            end_row = row_ptr[row + 1];
        }

        int sum = 0;
        for(unsigned int i = start_row; i < end_row; i++){
            sum += value[i] * d_in[index[i]];
        }
        d_out[row] = sum;
    }
}

int main(){
    const unsigned int MATRIX_BYTES = MATRIX_SIZE * sizeof(int);
    const unsigned int ELEMENT_BYTES = NON_ZERO_ELEMENTS * sizeof(int);

    // CSR definition
    int h_value[NON_ZERO_ELEMENTS] = {1, -3, 2, 5, -2, 7};
    int h_index[NON_ZERO_ELEMENTS] = {0, 2, 1, 0, 2, 3};
    int h_row_ptr[MATRIX_SIZE] = {0, 2, 3, 5};

    int* d_value;
    int* d_index;
    int* d_row_ptr;

    cudaMalloc((void **) &d_value, ELEMENT_BYTES);
    cudaMalloc((void **) &d_index, ELEMENT_BYTES);
    cudaMalloc((void **) &d_row_ptr, MATRIX_BYTES);

    cudaMemcpy(d_value, h_value, ELEMENT_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_index, h_index, ELEMENT_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr, h_row_ptr, MATRIX_BYTES, cudaMemcpyHostToDevice);

    // The vector
    int h_in[MATRIX_SIZE] = {1, 1, 1, 1};
    int h_out[MATRIX_SIZE];

    int* d_in;
    int* d_out;
    
    cudaMalloc((void **) &d_in, MATRIX_BYTES);
    cudaMalloc((void **) &d_out, MATRIX_BYTES);

    cudaMemcpy(d_in, h_in, MATRIX_BYTES, cudaMemcpyHostToDevice);
    spmv<<<NUM_BLOCKS, NUM_THREADS>>>(d_value, d_index, d_row_ptr, d_in, d_out);
    cudaMemcpy(h_out, d_out, MATRIX_BYTES, cudaMemcpyDeviceToHost);

    print_vector(h_out);

    cudaFree(d_value);
    cudaFree(d_index);
    cudaFree(d_row_ptr);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
