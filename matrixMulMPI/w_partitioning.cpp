#include <mpi.h>
#include <iostream>
#include <chrono>

using namespace std;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, total_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_rank);
    MPI_Barrier(MPI_COMM_WORLD);

    const int n = 8;
    const int local_n = n / total_rank; // Ensure n is divisible by total_rank

    // Allocate memory for local parts of matrices
    int *A = nullptr;
    int *B = new int[n * n];
    int *A_local = new int[local_n * n];
    int *A_output = new int[local_n * n];

    if (rank == 0) {
        A = new int[n * n];
        int count = 1;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[(i*n) + j] = count;
                count++;
            }
        }
        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                if (i == j)
                    B[(i*n) + j] = 1;
                else
                    B[(i*n) + j] = 0;
            }
        }
    }

    // Broadcast only the necessary portion of B
    MPI_Bcast(B, n * n, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Scatter(A, local_n * n, MPI_INT, A_local, local_n * n, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform matrix multiplication locally
    for (int i = 0; i < local_n; i++) {
        int row = rank * local_n + i;
        for (int j = 0; j < n; j++) {
            A_output[i * n + j] = 0;
            for (int k = 0; k < n; k++) {
                A_output[i * n + j] += A_local[i * n + k] * B[k * n + j];
            }
        }
    }

    // Gather and print results 
    if (rank == 0) {
        int *result = new int[n * n];
        MPI_Gather(A_output, local_n * n, MPI_INT, result, local_n * n, MPI_INT, 0, MPI_COMM_WORLD);
        cout << "Resultant Matrix A * B:" << endl;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                cout << result[i * n + j] << "\t";
            }
            cout << endl;
        }
        delete[] result;
    } else {
        MPI_Gather(A_output, local_n * n, MPI_INT, nullptr, 0, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        delete[] A;
    }
    delete[] B;
    delete[] A_local;
    delete[] A_output;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}