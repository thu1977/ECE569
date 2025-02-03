#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_rank == 0) {
        std::cout << "Hello from master process " << world_rank << std::endl;
        for (int i = 1; i < world_size; i++) {
            std::string message = "Hello from process " + std::to_string(i);
            MPI_Send(message.c_str(), message.length() + 1, MPI_CHAR, i, 0, MPI_COMM_WORLD);
        }
    } else {
        char message[100];
        MPI_Recv(message, 100, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Process " << world_rank << " received message: " << message << std::endl;
    }

    MPI_Finalize();
    return 0;
}
