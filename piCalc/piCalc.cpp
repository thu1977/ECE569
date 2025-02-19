#include <iostream>
#include <mpi.h>
#include <stdlib.h>
#include <time.h>


#define TOTAL_POINTS 1000000000

using namespace std;

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);
    int rank;
    int nproc;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double start_time = MPI_Wtime();

    srand(time(NULL) + rank);
    int local_points = TOTAL_POINTS / nproc;
    int local_hits = 0;
    int global_hits = 0;
    double x, y;
    double pi;

    for (int i = 0; i < local_points; i++) {
        x = (double)rand() / RAND_MAX;
        y = (double)rand() / RAND_MAX;
        if (x * x + y * y <= 1) {
            local_hits++;
        }
    }


    MPI_Reduce(&local_hits, &global_hits, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    double end_time = MPI_Wtime();

    // MPI_Reduce parameters: sendbuf, recvbuf, count, datatype, op, root, comm

    if (rank == 0) {
        pi = (double)global_hits / TOTAL_POINTS * 4;
        cout << "Pi is " << pi << endl;
        cout << "Time taken is " << end_time - start_time << endl;
    } else {
        cout << "Process " << rank << " has " << local_hits << " hits" << endl;
        cout << "Total time taken is " << end_time - start_time << endl;
    }


    MPI_Finalize();

    return 0;
}

// mpic++ piCalc.cpp -o piCalc 
// mpirun -np 6 piCalc

// results:
// tianyu@HuMac piCalc % mpirun -n 6 ./piCalc
// Process 3 has 130889673 hits
// Total time taken is 2.57205
// Process 4 has 130900816 hits
// Total time taken is 2.57425
// Process 2 has 130894744 hits
// Total time taken is 2.57761
// Process 5 has 130897597 hits
// Total time taken is 2.57924
// Pi is 3.14149
// Time taken is 2.57925
// Process 1 has 130897058 hits
// Total time taken is 2.57925