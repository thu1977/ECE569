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
    int n = 8;
    int A[n * n];
    int B[n * n];
    int A_input[n];
    int A_output[n];
    if (rank == 0) {
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
    // A * B = A; because B is an identity matrix that professor showed in one class
    MPI_Bcast(B, n * n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(A, n, MPI_INT, A_input, n, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A_input[j] * B[(k * n) + j];
            }
            A_output[j] = sum;
        }
    }
    cout << "Rank: " << rank << endl;
    for (int j = 0; j < n; j++) {
        cout << A_output[j] << "\t";
    }
    cout << endl;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}

/*
make
mpirun -np 4 ./gemm
Rank: 0
1       2       3       4       5       6       7       8
Rank: 1
9       10      11      12      13      14      15      16
Rank: 2
17      18      19      20      21      22      23      24
Rank: 3
25      26      27      28      29      30      31      32
 */