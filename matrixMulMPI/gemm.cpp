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
    int position = 0;
    if (rank == 0) {
        int count = 1;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[position] = count;
                count++;
                position++;
            }
        }
        count = 1;
        position = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                B[position] = count;
                count++;
                position++;
            }
        }
    }
    MPI_Bcast(B, n * n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(A, n, MPI_INT, A_input, n, MPI_INT, 0, MPI_COMM_WORLD);
    if(rank == 1){
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                cout << B[(i*n)+j] << "\t";
            }
            cout << endl;
        }
    }

    position = 0;
    if(rank == 1){
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int sum = 0;
                for (int k = 0; k < n; k++) {
                    cout << A_input[i] << " " << B[(k * n) + j] << endl;
//                    sum += A_input[i] * B[(k * n) + j];
                }
            }
            break;
            position++;
        }
    }
//    if(rank == 1){
//        for (int j = 0; j < n; j++) {
//            cout << A_output[j] << "\t";
//        }
//    }


    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}

//    auto start_time_chrono = std::chrono::high_resolution_clock::now();
//    auto start_time_mpi = MPI_Wtime();
//    std::cout << "Total ranks: " << total_rank << ", current rank: " << rank << std::endl;

//auto end_time_mpi = MPI_Wtime();
//auto end_time_chrono = std::chrono::high_resolution_clock::now();
//std::chrono::duration<double> elapsed_chrono = end_time_chrono - start_time_chrono;
//std::cout << "Chrono Time: " << (elapsed_chrono).count() << std::endl;
//std::cout << "MPI Time: " << (end_time_mpi - start_time_mpi) << std::endl;
