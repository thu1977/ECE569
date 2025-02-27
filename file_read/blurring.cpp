#include <iostream>
#include <mpi.h>
#include <fstream>
#include <cstring>
#include <vector>

// Read and Blur Out Raw Data with MPI, assign parameter to represent blurring effect


using namespace std;

const int width = 128;
const int height = 128;
const int pixel_size = 1; 
const int file_size = width * height * pixel_size;
const char* filename = "brain_128x128_uint8.raw";
const char* output_filename = "blurred_brain_128x128_uint8.raw";

void blurring(uint8_t* data, uint8_t* output, int width, int height, int radius) {
    memcpy(output, data, width * height * pixel_size);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int sum = 0;
            int count = 0;
            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    int nx = x + dx;
                    int ny = y + dy;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        sum += data[ny * width + nx];
                        count++;
                    }
                }
            }
            output[y * width + x] = sum / count;
        }
    }
}

int main( int argc, char **argv ) {
    MPI_Init( &argc, &argv );
    int rank, total_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_rank);

    int radius = 2; // Blurring radius

    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
    

    // Calculate the number of bytes each process will read
    int local_size = file_size / total_rank;
    int offset = rank * local_size;

    // Allocate memory for the local data
    uint8_t* local_data = new uint8_t[local_size];

    // Read the data from the file
    MPI_File_read_at(file, offset, local_data, local_size, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
    MPI_File_close(&file);

    // Print the local data
    // for (int i = 0; i < local_size; i++) {
    //     cout << (int)local_data[i] << " ";
    //     if ((i + 1) % width == 0) {
    //         cout << endl;
    //     }
    // }

    vector<uint8_t> output_data(file_size);
    blurring(local_data, output_data.data() + offset, width, height, radius);

    MPI_File output_file;
    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file, offset, output_data.data() + offset, local_size, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);

    delete[] local_data;
    MPI_Finalize();

    return 0;
}