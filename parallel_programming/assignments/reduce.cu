/* 
Implementing parallell plus reduce in CUDA.
*/

#include <stdio.h>
#define NUM_THREADS 16
#define NUM_BLOCKS 8

unsigned int serial_reduce(unsigned int* array, const unsigned int size){
    unsigned int sum = 0;

    for(int i = 0; i < size; i++){
        sum += array[i];
    }

    return sum;
}

__global__ void reduce(unsigned int* d_in, unsigned int* d_out){
    
    unsigned int local_idx = threadIdx.x;
    unsigned int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int num_threads = blockDim.x;

    extern __shared__ unsigned int shared_array[];
    shared_array[local_idx] = d_in[global_idx];
    __syncthreads();

    for(unsigned int i = 1; i < num_threads; i *= 2){
        if(local_idx % 2 * i == 0){
            shared_array[local_idx] = shared_array[local_idx] + shared_array[local_idx + i];
        }
        __syncthreads();
    }

    d_out[blockIdx.x] = shared_array[0];
}

int main(){
    const unsigned int NUM_ELEMENTS = NUM_THREADS * NUM_BLOCKS;
    const unsigned int IN_BYTES = NUM_ELEMENTS * sizeof(int);
    const unsigned int OUT_BYTES = NUM_BLOCKS * sizeof(int);
    
    unsigned int h_in [NUM_ELEMENTS];
    for(int i = 0; i < NUM_ELEMENTS; i++){
        h_in[i] = i;
    }

    unsigned int h_out [NUM_BLOCKS];

    unsigned int* d_in;
    unsigned int* d_out;

    cudaMalloc((void **) &d_in, IN_BYTES);
    cudaMalloc((void **) &d_out, OUT_BYTES);

    cudaMemcpy(d_in, h_in, IN_BYTES, cudaMemcpyHostToDevice);
    reduce<<<NUM_BLOCKS, NUM_THREADS, NUM_THREADS * sizeof(int)>>>(d_in, d_out);
    cudaMemcpy(h_out, d_out, OUT_BYTES, cudaMemcpyDeviceToHost);

    printf("True: %d \n", serial_reduce(h_in, NUM_ELEMENTS));

    // Doing a final serial reduce since output is of size NUM_BLOCKS
    printf("Output: %d", serial_reduce(h_out, NUM_BLOCKS)); 

    return 0;
}
