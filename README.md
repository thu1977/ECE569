ECE/CS 569 High-Performance Processors and Systems

## Running MPI Program using Makefile
- Compile:
```shell
cd makefileExample
make all
mpicxx -O3 -o simple simple.cpp
➜  makefileExample mpirun -np 4 ./simple
```
- Execute:
```shell
mpirun -np 4 ./simple
Total ranks: 4, current rank: 0
Chrono Time: 7.7422e-05
MPI Time: 6.9533e-05
Total ranks: 4, current rank: 1
Chrono Time: 7.3499e-05
MPI Time: 6.7903e-05
Total ranks: 4, current rank: 2
Chrono Time: 7.2819e-05
MPI Time: 6.7041e-05
Total ranks: 4, current rank: 3
Chrono Time: 7.3504e-05
MPI Time: 7.2274e-05
```
- Clean:
```shell
make clean
rm -f simple
```