#include <chrono>
#include <iostream>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <mpi.h>

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


__global__
void countElem(int n, int find, int *data, int *d_result)
{
  int batch_size = n/gridDim.x;
  int idx = threadIdx.x + batch_size*blockIdx.x;

  int batch_count = 0;

  for(int i = idx; i<batch_size*(blockIdx.x+1); i+=blockDim.x){
    if(data[i] == find){
      batch_count++;
    }
  }
 
  atomicAdd(&d_result[blockIdx.x], batch_count);
}

int countElemCPU(int n, int find, int *data){
  unsigned int total_count = 0;
  for(int i = 0; i<n; i++){
    if(data[i] == find){
      total_count++;
    }
  }
  return total_count;
}

__global__
void initialize(int* data, int N){

  int idx = threadIdx.x+blockDim.x*blockIdx.x;

  for(int i = idx; i <  N; i += blockDim.x){
    data[i] = 50;
  }
}

int main(void)
{

  auto clock = std::chrono::high_resolution_clock();
  auto start = clock.now();

  int deviceCount;
  HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));

  if(deviceCount != 4){
    std::cerr << "GPUs available are: " << deviceCount << std::endl;
    if(deviceCount != 1){
      std::cerr << "Device count not suitable." << std::endl;
      return 1;
    }
  }

  int N = 1<<28;
  int blockSize = 256;
  int numBlocks = 8;

  int batchN = N/deviceCount;

  for(int gpuID = 0; gpuID<deviceCount; gpuID++){
    int *d_result = new int[numBlocks];
    int *result = new int[numBlocks];
    int *data = new int[batchN];
    int *d_data = new int[batchN];

    //In this case we initialize the data so it's not necessary to divide the data

    HANDLE_ERROR(cudaSetDevice(gpuID));
    HANDLE_ERROR(cudaMalloc(&d_data, batchN*sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&d_result, numBlocks*sizeof(int)));

    initialize<<<numBlocks, blockSize>>>(d_data, batchN);
  
    countElem<<<numBlocks, blockSize>>>(batchN, 50,d_data, d_result);

    // MPI non-blocking send of the result GPU-0 device (MPI cuda-aware)
    // per ora no

    HANDLE_ERROR(cudaMemcpy(result, d_result, numBlocks*sizeof(int), cudaMemcpyDeviceToHost));

    // MPI non-blocking send of the result to GPU-0 host



    int final_count = 0;
    for(int i = 0; i<numBlocks; i++){
      final_count += result[i];
    }

    std::cout << "Element count: " << N << std::endl;  
    std::cout << "Device variable value: " << final_count <<std::endl;

    // Free memory
    HANDLE_ERROR(cudaFree(d_result));
    HANDLE_ERROR(cudaFree(d_data));

    auto end = clock.now(); 
    long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    std::cout << "Time: "<<  (float) microseconds/1000 << "ms" << std::endl;
  }

  HANDLE_ERROR(cudaDeviceSynchronize());

  //MPI blocking recv to receive results and merge them

  return 0;
}