/* 
Implementing 5-point 2D stencil in CUDA.
*/

#include <stdio.h>
#define MATRIX_SIZE 4
#define BLOCK_WIDTH 2

void print_matrix(float* array){
    for(unsigned int i = 0; i < MATRIX_SIZE; i++){
        for(unsigned int j = 0; j < MATRIX_SIZE; j++){
            printf("%.3f ", array[j + i * MATRIX_SIZE]);
        }
        printf("\n");
    }
}

__device__ int get_index(int row, int col){
    return col + row*MATRIX_SIZE;
}

__global__ void stencil(float* d_in, float* d_out){
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;

    if(row >= MATRIX_SIZE || col >= MATRIX_SIZE){
        return;
    }

    float sum = d_in[get_index(row, col)];
    unsigned int num_neigbors = 1;

    // Checking boundaries
    if(col - 1 >= 0){
        sum += d_in[get_index(row, col - 1)];
        num_neigbors++;
    }

    if(col + 1 < MATRIX_SIZE){
        sum += d_in[get_index(row, col + 1)];
        num_neigbors++;
    }

    if(row - 1 >= 0){
        sum += d_in[get_index(row - 1, col)];
        num_neigbors++;
    }

    if(row + 1 < MATRIX_SIZE){
        sum += d_in[get_index(row + 1, col)];
        num_neigbors++;
    }
   
    sum /= ((float) num_neigbors);
    d_out[get_index(row, col)] = sum;
}

int main(){
    const unsigned int NUM_ELEMENTS = MATRIX_SIZE * MATRIX_SIZE;
    const unsigned int BYTES = NUM_ELEMENTS * sizeof(float);

    float h_in[NUM_ELEMENTS];

    for(unsigned int i = 0; i < NUM_ELEMENTS; i++){
        h_in[i] = (float) i;
    }

    float h_out[NUM_ELEMENTS];

    float* d_in;
    float* d_out;
    
    cudaMalloc((void **) &d_in, BYTES);
    cudaMalloc((void **) &d_out, BYTES);
    
    const dim3 block_size(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    const unsigned int row_blocks = (unsigned int) (MATRIX_SIZE / BLOCK_WIDTH + 1);
    const unsigned int col_blocks = (unsigned int) (MATRIX_SIZE / BLOCK_WIDTH + 1);
    const dim3 grid_size(row_blocks, col_blocks, 1);

    cudaMemcpy(d_in, h_in, BYTES, cudaMemcpyHostToDevice);
    stencil<<<grid_size, block_size>>>(d_in, d_out);
    cudaMemcpy(h_out, d_out, BYTES, cudaMemcpyDeviceToHost);

    printf("Matrix: \n");
    print_matrix(h_in);
    printf("\n");

    printf("Matrix after stencil: \n");
    print_matrix(h_out);

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
