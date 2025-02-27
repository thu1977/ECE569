#include <iostream>
#include <mpi.h>
#include <vector>

void blur(uint8_t* local_data, uint8_t* output_data, int width, int height, int local_size, int blur_radius) {
    int row_start = 0;
    int row_end = height - 1;
    int col_start = 0;
    int col_end = width - 1;

    int kernel_size = 2 * blur_radius + 1; // Ensure the kernel size is odd

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int sum = 0;
            int count = 0;

            for (int k = -blur_radius; k <= blur_radius; k++) {
                for (int l = -blur_radius; l <= blur_radius; l++) {
                    int ni = i + k;
                    int nj = j + l;

                    // Ensure the neighbor pixel is within bounds
                    if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
                        sum += local_data[ni * width + nj];
                        count++;
                    }
                }
            }

            // Calculate the average of the neighbors
            output_data[i * width + j] = sum / count;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, total_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_rank);

    const char* filename = "brain_128x128_uint8.raw";
    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);

    const int width = 128;
    const int height = 128;
    const int pixel_size = 1;
    const int file_size = width * height * pixel_size;

    int local_size = file_size / total_rank;
    int offset = rank * local_size;

    uint8_t* local_data = new uint8_t[local_size];
    uint8_t* output_data = new uint8_t[local_size];

    MPI_File_read_at(file, offset, local_data, local_size, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
    MPI_File_close(&file);

    // Set blur intensity (radius)
    int blur_radius = 0;  
    
    blur(local_data, output_data, width, height, local_size, blur_radius);

    const char* output_filename = "blurred_brain_128x128_uint8.raw";
    MPI_File output_file;
    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file, offset, output_data, local_size, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);

    delete[] local_data;
    delete[] output_data;

    MPI_Finalize();

    return 0;
}
