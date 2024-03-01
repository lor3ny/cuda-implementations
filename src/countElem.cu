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
void countElem(int n, int find, int *data, int *result)
{
  int batch_size = n/blockDim.x;
  int start_index = threadIdx.x * batch_size;

  int batch_count = 0;

  for(int i = start_index; i<start_index+batch_size; i++){
    if(data[i] == find){
      batch_count++;
    }
  }

  result[threadIdx.x] = batch_count;
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

  int *result = new int[blockSize*numBlocks];
  int *x = new int[N];

  // Allocate Unified Memory – accessible from CPU or GPU
  HANDLE_ERROR(cudaMallocManaged(&x, N*sizeof(int)));
  HANDLE_ERROR(cudaMallocManaged(&result, blockSize*numBlocks*sizeof(int)));

  initialize(x, N);
  
  countElem<<<numBlocks, blockSize>>>(N, 50,x, result);

  // Wait for GPU to finish before accessing on host
  HANDLE_ERROR(cudaDeviceSynchronize());

  int final_count = 0;
  for(int i = 0; i<blockSize; i++){
    final_count += result[i];
  }

  std::cout << "Element count: " << N << std::endl;  
  std::cout << "Device variable value: " << final_count <<std::endl;

  // Free memory
  HANDLE_ERROR(cudaFree(result));
  HANDLE_ERROR(cudaFree(x));
  
  return 0;
}