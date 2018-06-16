/* 
Implementing Radix sort in CUDA.
*/

#include <stdio.h>
#include <stdlib.h>
#define NUM_ELEMENTS 16

__device__ void partition_by_bit(unsigned int* values, unsigned int bit);

__global__ void radix_sort(unsigned int* d_array){
    for(int bit = 0; bit < 32; bit++){
        partition_by_bit(d_array, bit);
        __syncthreads();
    }
}

__device__ unsigned int plus_scan(unsigned int* bit_array){
    unsigned int idx = threadIdx.x;
    unsigned int size = blockDim.x;

    for(int offset = 1; offset < size; offset *= 2){
        unsigned int array_offset;

        if (idx >= offset){
            array_offset = bit_array[idx - offset];
        }
        __syncthreads();

        if(idx >= offset){
            bit_array[idx] = array_offset + bit_array[idx];
        }
  
        __syncthreads();
    }

    return bit_array[idx];
}

__device__ void partition_by_bit(unsigned int* values, unsigned int bit){
    unsigned int idx = threadIdx.x;
    unsigned int size = blockDim.x;
    unsigned int x_i = values[idx];
    unsigned int p_i = (x_i >> bit) & 1;

    values[idx] = p_i;  

    __syncthreads();

    unsigned int scan_val = plus_scan(values);
    unsigned int total = size - values[size - 1];

    __syncthreads();

    if (p_i){
        values[scan_val - 1 + total] = x_i;
    }else{
        values[idx - scan_val] = x_i;
    }       
}

int main(){
    const unsigned int BYTES = NUM_ELEMENTS*sizeof(int);
    
    unsigned int h_in[NUM_ELEMENTS];
    unsigned int h_out [NUM_ELEMENTS];

    for(int i = 0; i < NUM_ELEMENTS; i++){
        h_in[i] = rand() % 100; // Generating random numbers between 0 and 99
    }

    unsigned int* d_array;
    cudaMalloc((void **) &d_array, BYTES);

    cudaMemcpy(d_array, h_in, BYTES, cudaMemcpyHostToDevice);
    radix_sort<<<1, NUM_ELEMENTS>>>(d_array);
    cudaMemcpy(h_out, d_array, BYTES, cudaMemcpyDeviceToHost);

    printf("Unsorted: \n");
    for(int i = 0; i < NUM_ELEMENTS; i++){
        printf("%d ", h_in[i]);
    }
    printf("\n");

    printf("Sorted: \n");
    for(int i = 0; i < NUM_ELEMENTS; i++){
        printf("%d ", h_out[i]);
    }

    return 0;
}
