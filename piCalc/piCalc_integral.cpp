#include <iostream>
#include <mpi.h>
#include <cmath>

#define NUM_INTERVALS 1000000000  

using namespace std;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, nproc;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double start_time = MPI_Wtime();

    double local_sum = 0.0;
    double global_sum = 0.0;
    double width = 1.0 / NUM_INTERVALS;  // Width of each interval
    
    for (int i = rank; i < NUM_INTERVALS; i += nproc) {
        double x = (i + 0.5) * width;  // height of each rectangle
        local_sum += 4.0 / (1.0 + x * x);
    }

    local_sum *= width;  // local integral

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    double end_time = MPI_Wtime();
    double local_time = end_time - start_time;
    double max_time;

    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Pi is " << global_sum << endl;
        cout << "Max time taken is " << max_time << " seconds" << endl;
    } else {
        cout << "Process " << rank << " finished in " << local_time << " seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}

// mpic++ piCalc_integral.cpp -o piCalc_integral
// mpirun -np 6 piCalc_integral

// rusults:
// tianyu@HuMac piCalc % mpirun -np 6 piCalc_integral
// Process 5 finished in 0.312977 seconds
// Process 4 finished in 0.313102 seconds
// Process 2 finished in 0.314654 seconds
// Process 3 finished in 0.319201 seconds
// Pi is 3.14159
// Max time taken is 0.319223 seconds
// Process 1 finished in 0.319223 seconds