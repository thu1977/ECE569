#include <iostream>
#include <mpi.h>

// Read and Print Out Raw Data with MPI

using namespace std;

int main( int argc, char **argv ) {
    MPI_Init( &argc, &argv );
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

    MPI_File_read_at(file, offset, local_data, local_size, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);


    MPI_File_close(&file);

    for (int i = 0; i < local_size; i++) {
        cout << (int)local_data[i] << " ";
        if ((i + 1) % width == 0) {
            cout << endl;
        }
    }

    delete[] local_data;
    MPI_Finalize();

    return 0;
} 