/* 
Implementing inclusive Hillis & Steele plus scan in CUDA.
*/

#include <stdio.h>
#define NUM_THREADS 16

void serial_scan(unsigned int* in_array, unsigned int* out_array, const unsigned int size){
    for(unsigned int i = 0; i < size; i++){
        unsigned int sum = 0;

        for(unsigned int j = 0; j <= i; j++){
            sum += in_array[j];
        }
        out_array[i] = sum;
    }
}

__global__ void scan(unsigned int* d_in){
    unsigned int idx = threadIdx.x;
    unsigned const int num_threads = blockDim.x;

    for(unsigned int i = 1; i < num_threads; i *= 2){
            if(idx >= i){
                d_in[idx] += d_in[idx - i];
            }
            __syncthreads();
        }
}

int main(){
    const unsigned int BYTES = NUM_THREADS * sizeof(int);
    
    unsigned int h_in [NUM_THREADS];
    for(unsigned int i = 0; i < NUM_THREADS; i++){
        h_in[i] = i + 1;
    }
    
    unsigned int h_out [NUM_THREADS];

    unsigned int* d_in;
    unsigned int* d_out;

    cudaMalloc((void **) &d_in, BYTES);
    cudaMalloc((void **) &d_out, BYTES);

    cudaMemcpy(d_in, h_in, BYTES, cudaMemcpyHostToDevice);
    scan<<<1, NUM_THREADS>>>(d_in);
    cudaMemcpy(h_out, d_in, BYTES, cudaMemcpyDeviceToHost);

    unsigned int true_output[NUM_THREADS];
    serial_scan(h_in, true_output, NUM_THREADS);

    printf("True: \n");
    for(unsigned int i = 0; i < NUM_THREADS; i++){
        printf("%d ", true_output[i]);
    }
    printf("\n");

    printf("Output: \n");
    for(unsigned int i = 0; i < NUM_THREADS; i++){
        printf("%d ", h_out[i]);
    }

    return 0;
}
