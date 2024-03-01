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

void initialize(int*& data, int& N){

  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<> distribution(0, 1000);

  for (int i = 0; i < N; i++) {
    int random_number= distribution(gen);
    data[i] = random_number;
  }
}

int main(void)
{
  int N = 1<<20;
  int blockSize = 256;
  int numBlocks = 1;

  int *d_result = new int[blockSize*numBlocks];
  int *result = new int[blockSize*numBlocks];
  int *x = new int[N];

  initialize(x, N);

  // Allocate Unified Memory – accessible from CPU or GPU
  HANDLE_ERROR(cudaMalloc(&x, N*sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&d_result, blockSize*numBlocks*sizeof(int)));
  
  countElem<<<numBlocks, blockSize>>>(N, 50,x, d_result);

  // Wait for GPU to finish before accessing on host
  HANDLE_ERROR(cudaDeviceSynchronize());

  HANDLE_ERROR(cudaMemcpy(result, d_result, blockSize*numBlocks*sizeof(int), cudaMemcpyDeviceToHost));

  int final_count = 0;
  for(int i = 0; i<blockSize; i++){
    final_count += result[i];
  }

  std::cout << "Element count: " << N << std::endl;  
  std::cout << "Device variable value: " << final_count <<std::endl;

  // Free memory
  HANDLE_ERROR(cudaFree(d_result));
  HANDLE_ERROR(cudaFree(x));
  
  return 0;
}