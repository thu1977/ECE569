#include <mpi.h>
#include <iostream>
#include <chrono>


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, total_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_rank);
    MPI_Barrier(MPI_COMM_WORLD);
    auto start_time_chrono = std::chrono::high_resolution_clock::now();
    auto start_time_mpi = MPI_Wtime();
    std::cout << "Total ranks: " << total_rank << ", current rank: " << rank << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time_mpi = MPI_Wtime();
    auto end_time_chrono = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_chrono = end_time_chrono - start_time_chrono;
    std::cout << "Chrono Time: " << (elapsed_chrono).count() << std::endl;
    std::cout << "MPI Time: " << (end_time_mpi - start_time_mpi) << std::endl;
    MPI_Finalize();
}