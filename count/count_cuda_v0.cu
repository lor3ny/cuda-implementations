#include <chrono>
#include <ctime>
#include <iostream>
#include <math.h>
#include <random>

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
  int batch_size = n/blockDim.x;
  int start_index = threadIdx.x * batch_size;

  int batch_count = 0;

  for(int i = start_index; i<start_index+batch_size; i++){
    if(data[i] == find){
      batch_count++;
    }
  }

  d_result[threadIdx.x] = batch_count;
  //atomicAdd(result, batch_count);
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

  int N = 1<<28;
  int blockSize = 256;
  int numBlocks = 1;

  int *d_result = new int[blockSize*numBlocks];
  int *d_data = new int[N];

  // Allocate Unified Memory – accessible from CPU or GPU
  HANDLE_ERROR(cudaMallocManaged(&d_data, N*sizeof(int)));
  HANDLE_ERROR(cudaMallocManaged(&d_result, blockSize*numBlocks*sizeof(int)));

  initialize<<<8, 256>>>(d_data, N);

  cudaMemcpy(d_data, d_data, N * sizeof(int), cudaMemcpyHostToDevice);
  
  countElem<<<numBlocks, blockSize>>>(N, 50,d_data, d_result);

  // Wait for GPU to finish before accessing on host
  HANDLE_ERROR(cudaDeviceSynchronize());

  HANDLE_ERROR(cudaMemcpy(d_result, d_result, blockSize*numBlocks*sizeof(int), cudaMemcpyDeviceToHost));

  int final_count = 0;
  for(int i = 0; i<blockSize; i++){
    final_count += d_result[i];
  }

  std::cout << "Element count: " << N << std::endl;  
  std::cout << "Device variable value: " << final_count <<std::endl;

  // Free memory
  HANDLE_ERROR(cudaFree(d_result));
  HANDLE_ERROR(cudaFree(d_data));

  auto end = clock.now(); 
  long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
  std::cout << "Time: "<<  (float) microseconds/1000 << "ms" << std::endl;

  return 0;
}