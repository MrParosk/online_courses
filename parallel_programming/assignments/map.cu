/* 
Implementing map in CUDA. It squares each element in a matrix.
*/

#include <stdio.h>
#define BLOCK_WIDTH 2
#define MATRIX_SIZE 4

void print_matrix(unsigned int* array){
    for(unsigned int i = 0; i < MATRIX_SIZE; i++){
        for(unsigned int j = 0; j < MATRIX_SIZE; j++){
            printf("%03d ", array[j + i * MATRIX_SIZE]);
        }
        printf("\n");
    }
}

__device__ unsigned int square(unsigned int x){
    return x*x;
}

__global__ void map(unsigned int* d_out, unsigned int* d_in){
    unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int col = threadIdx.y + blockIdx.y * blockDim.y;

    if (row >= MATRIX_SIZE || col >= MATRIX_SIZE){
        return;
    }

    unsigned int idx = col + row * MATRIX_SIZE;
    d_out[idx] = square(d_in[idx]);
}

int main(){
    const unsigned int NUM_ELEMENTS = MATRIX_SIZE * MATRIX_SIZE;
    const unsigned int BYTES = NUM_ELEMENTS * sizeof(int);
    
    unsigned int h_in[NUM_ELEMENTS];

    for(unsigned int i = 0; i < NUM_ELEMENTS; i++){
        h_in[i] = i;
    }

    unsigned int h_out[NUM_ELEMENTS];

    unsigned int* d_in;
    unsigned int* d_out;

    cudaMalloc((void **) &d_in, BYTES);
    cudaMalloc((void **) &d_out, BYTES);

    const dim3 block_size(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    const unsigned int row_blocks = (unsigned int) (MATRIX_SIZE / BLOCK_WIDTH + 1);
    const unsigned int col_blocks = (unsigned int) (MATRIX_SIZE / BLOCK_WIDTH + 1);
    const dim3 grid_size(row_blocks, col_blocks, 1);

    cudaMemcpy(d_in, h_in, BYTES, cudaMemcpyHostToDevice);
    map<<<grid_size, block_size>>>(d_out, d_in);
    cudaMemcpy(h_out, d_out, BYTES, cudaMemcpyDeviceToHost);

    printf("Matrix before squaring: \n");
    print_matrix(h_in);
    printf("\n");

    printf("Matrix after squaring: \n");
    print_matrix(h_out);

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
