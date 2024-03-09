# cuda-parallel_search
Parallel programming with CUDA, MPI and OpenMP for HPC applications.


## Compiling tools


**CUDA simple compiling**
```console
nvcc -o name file.cu
```

**MPI compile for C++**
```console
mpicxx -o name file.cpp
```
or
```console
mpic++ -o name file.cpp
```

**MPI execution**
```console
mpirun -np procs_count ./name
```

**MPI cuda-aware**
*Work in progress*


## Dependencies

[CUDA-12](https://developer.nvidia.com/cuda-toolkit)
[OpenMPI-5.0.2](https://www.open-mpi.org/software/ompi/v5.0/)
