# include <cuda_runtime.h>
# include <stdio.h>

using namespace std;

__global__ void helloFromGPU(){
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main(){
    helloFromGPU<<<1, 10>>>(); // Launching kernel with 1 block and 10 threads per block
    cudaDeviceSynchronize(); // Wait for GPU to finish before exiting
    return 0;
}

// Compile with: nvcc helloCuda.cu -o helloCuda
// Run with: ./helloCuda