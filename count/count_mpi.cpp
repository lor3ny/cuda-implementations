#include <mpi.h>
#include <iostream>
#include <ostream>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;

inline void createBuff(int* buff, const char* filename, int count){
  for(int i = 0; i<count; i++){
    buff[i] = 50;
  }
}

int main(int argc, char *argv[]){

    MPI_Init(&argc, &argv);

    int N = 1<<20;
    int numProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    int batchLength = N/numProcs;
    int target = 50;
    string filename = "profiles20.csv";

    // Set the current process
    int task_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &task_id);

    int subBuff[batchLength];

    auto startTime = MPI_Wtime();

    if(task_id == 0){

        int startIndex = task_id*batchLength;
        int endIndex = startIndex+batchLength;

        // Import data
        int buff[N];
        createBuff(buff, filename.c_str(), N);
        cout << sizeof(buff)/sizeof(int) << endl;

        // Scatter data
        MPI_Scatter(buff, batchLength, MPI_INT, subBuff, batchLength, MPI_INT, 0, MPI_COMM_WORLD);

        // Computation

        int counter = 0;
        for(int i = 0; i<batchLength; i++){
            if(buff[i] == target){
                counter++;
            }
        }

        // Allreduce

        int result;
        MPI_Allreduce(&counter, &result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        cout << task_id << " - partial: " << counter << " - result: " << result << endl;

    } else {

        // Scatter
        MPI_Scatter(nullptr, 0, MPI_INT, &subBuff, batchLength, MPI_INT, 0, MPI_COMM_WORLD);


        // Computation
        int counter = 0;
        for(int i = 0; i<(sizeof(subBuff)/sizeof(int)); i++){
            if(subBuff[i] == target){
                counter++;
            }
        }

        // Allreduce

        int result;
        MPI_Allreduce(&counter, &result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        cout << task_id << " - partial: " << counter << " - result: " << result << endl; 
    }

    auto endtime = MPI_Wtime();
    cout << endtime-startTime << endl;

    MPI_Finalize();
}