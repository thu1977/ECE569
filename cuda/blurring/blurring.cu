#include <stdio.h>
#include <cuda_runtime.h>

#define WIDTH 128
#define HEIGHT 128

__global__ void blur_kernel(unsigned char* input, unsigned char* output, int width, int height, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // column
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // row

    if (x >= width || y >= height) return;  // out of bounds

    int half_k = kernel_size / 2;
    int sum = 0;
    int count = 0;

    for (int ky = -half_k; ky <= half_k; ++ky) {
        for (int kx = -half_k; kx <= half_k; ++kx) {
            int nx = x + kx;
            int ny = y + ky;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                sum += input[ny * width + nx];
                count++;
            }
        }
    }

    output[y * width + x] = sum / count;
}

__host__ int main() {
    const int size = WIDTH * HEIGHT * sizeof(unsigned char);
    unsigned char *h_input = new unsigned char[WIDTH * HEIGHT];
    unsigned char *h_output = new unsigned char[WIDTH * HEIGHT];

    // Read RAW file
    FILE* f_in = fopen("brain_128x128_uint8.raw", "rb");
    fread(h_input, sizeof(unsigned char), WIDTH * HEIGHT, f_in);
    fclose(f_in);

    unsigned char *d_input, *d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Kernel configuration, then call Kernel the Global
    dim3 blockDim(16, 16);
    dim3 gridDim((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    int kernel_size = 5;

    blur_kernel<<<gridDim, blockDim>>>(d_input, d_output, WIDTH, HEIGHT, kernel_size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Write to output file
    FILE* f_out = fopen("blurred_output.raw", "wb");
    fwrite(h_output, sizeof(unsigned char), WIDTH * HEIGHT, f_out);
    fclose(f_out);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;

    printf("Blurring completed with kernel size = %d. Output saved to blurred_output.raw\n", kernel_size);
    return 0;
}
