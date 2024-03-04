#include <chrono>
#include <cstdint>
#include <iostream>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>

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

  atomicAdd(d_result, batch_count);
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
void initialize(int* data, int N, int *d_result){

  *d_result = 0;

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

  int *d_result;
  int *result = new int;
  int *data = new int[N];
  int *d_data = new int[N];

  HANDLE_ERROR(cudaMalloc(&d_data, N*sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&d_result, sizeof(int)));

  initialize<<<8, 256>>>(d_data, N, d_result);
  
  countElem<<<numBlocks, blockSize>>>(N, 50,d_data, d_result);

  HANDLE_ERROR(cudaDeviceSynchronize());

  HANDLE_ERROR(cudaMemcpy(result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

  std::cout << "Element count: " << N << std::endl;  
  std::cout << "Device variable value: " << *result <<std::endl;

  // Free memory
  HANDLE_ERROR(cudaFree(d_result));
  HANDLE_ERROR(cudaFree(d_data));

  auto end = clock.now(); 
  long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
  std::cout << "Time: "<<  (float) microseconds/1000 << "ms" << std::endl;

  return 0;
}