#include <algorithm>
#include <cmath>
#include <mpi.h>
#include <ostream>
#include <random>
#include <iostream>
#include <set>


#define N 1<<10


using namespace std;



int main(int argc, char *argv[]){
    
    MPI_Init(&argc, &argv);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int id;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    int squarel = 2; //so the range is between 0<=x<=2 and 0<=y<=2
    int circler = 1; //so the range is ?


    if(id==0){
        random_device rd;
        mt19937 gen(rd());

        set<float> unique_numbers; //No repetitions
        
        int n = N;
        int batchSize = (N)/(size-1);
        MPI_Bcast(&batchSize, 1, MPI_INT, 0, MPI_COMM_WORLD);


        // Con Iprobe e' possibile rendere questo processo lavoratore

        MPI_Status status;
        int counter = 0;
        do {
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            int ret;
            MPI_Recv(&ret, 1, MPI_FLOAT, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, &status);

            if(ret == -1){
                counter++;
            } else {
                unique_numbers.insert(ret);
            }
            
        } while (counter < size-1);
    
        cout << "I'm " << id << endl;

        cout << unique_numbers.size() << endl;

        MPI_Barrier(MPI_COMM_WORLD);
        
    } else {

        random_device rd;
        mt19937 gen(rd());

        int batchN;

        MPI_Bcast(&batchN, 1, MPI_INT, 0, MPI_COMM_WORLD);

        int counter = 0;
        while(counter < batchN){
            float rand_n = ((float) uniform_int_distribution<int>(0,2000)(gen));
            MPI_Send(&rand_n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            counter++;
        }

        int ending = -1;
        MPI_Send(&ending, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

        cout << "I'm " << id << endl;

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Printing;

    MPI_Finalize();

}