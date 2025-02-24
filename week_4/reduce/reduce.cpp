#include <iostream>
#include <mpi.h>
#include <stdlib.h>
#include <time.h>


using namespace std;

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);
    int rank;
    int nproc;

    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    srand(time(NULL) + rank);
    int local_variable = rand() % 100;
    int global_max = local_variable;
    MPI_Status status;

    cout << "Process " << rank << " has local variable " << local_variable << endl;

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 1; i < nproc; i++) {
            MPI_Recv(&local_variable, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
            if (local_variable > global_max) {
                global_max = local_variable;
            }
        }
    } else {
        MPI_Send(&local_variable, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    // MPI_Reduce(&local_variable, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    cout << "Process " << rank << " has global max " << global_max << endl;

    MPI_Finalize();

    return 0;
}


// Compile with: mpic++ reduce.cpp -o reduce 
// Run with: mpirun -np 4 ./reduce