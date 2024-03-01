#include <iostream>
#include <math.h>
#include <random>
#include <chrono>


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


  // DATA
  int N = 1<<20;
  int blockSize = 256;
  int numBlocks = 1;
  int *data = new int[N];
  int *d_result = new int[blockSize];
  int *result = new int[blockSize];
  bool is_gpu = true;
  auto clock = std::chrono::high_resolution_clock();

  initialize(data, N);

  cudaMalloc(&data, N*sizeof(int));
  cudaMalloc(&d_result, blockSize*sizeof(int));

  auto start = clock.now();

  if(!is_gpu){

    std::cout << "--- CPU computation ---" << std::endl;

    int res = countElemCPU(N, 50, data);

    std::cout << "Element count: " << N << std::endl;  
    std::cout << "Device variable value: " << res <<std::endl;

  } else {

    std::cout << "--- GPU computation ---" << std::endl;
  
    countElem<<<numBlocks, blockSize>>>(N, 50, data, d_result);

    cudaDeviceSynchronize();

    cudaMemcpy(result, d_result, blockSize*sizeof(int), cudaMemcpyDeviceToHost);

    int res = 0;
    for(int i = 0; i<blockSize; i++){
      res += result[i];
    }

    std::cout << "Element count: " << N << std::endl;  
    std::cout << "Device variable value: " << res <<std::endl;

    cudaFree(d_result);
    cudaFree(data);
  }

  auto end = clock.now();

  // GPU
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Time: " << duration.count() << "ms" << std::endl;

  
  return 0;
}