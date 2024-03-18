#include <algorithm>
#include <cmath>
#include <mpi.h>
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

        set<float> unique_numbers; //No repetitions
        
        int n = N;
        int batchSize = (N)/size;
        MPI_Bcast(&batchSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

        float value;
        while(unique_numbers.size() < N){
            MPI_Status tmpStatus;
            
            cout << "start " << (N)-unique_numbers.size() << endl;

            MPI_Recv(&value, 1, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &tmpStatus);

            cout << "end" << endl; 

            unique_numbers.insert(value);

            if(value<0){
                int remains;
                if(size>((N)-unique_numbers.size())){
                    remains = size;
                } else {
                    remains = ((N)-unique_numbers.size())/size;
                }
                MPI_Send(&remains, 1, MPI_INT, tmpStatus.MPI_SOURCE, 0, MPI_COMM_WORLD);
            }
        }
        cout << id << endl; 


        for(int i = 0; i<size; i++){
            MPI_Status tmpStatus;
            float ret;
            MPI_Recv(&ret, 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &tmpStatus);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        
        
        cout << "Result: " << unique_numbers.size() << endl;
        /*
        for (set<float>::iterator i = unique_numbers.begin(); i != unique_numbers.end(); i++) {
            float element = *i;
            cout << element << ",";
        }
        */

    } else {

        random_device rd;
        mt19937 gen(rd());

        int bound;

        MPI_Bcast(&bound, 1, MPI_INT, 0, MPI_COMM_WORLD);

        cout << "RECEVEID: " << bound << endl; 

        int counter = 0;
        while(counter < bound){
            for(int i = 0; i<bound; ++i){
                float rand_n = ((float) uniform_int_distribution<int>(0,2000)(gen))/1000;
                MPI_Send(&rand_n, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
            }

            MPI_Status status;
            float askValue= -0.1;

            cout << "REQUEST FROM: " << id << endl; 

            MPI_Send(&askValue, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
            bound = 0;
            counter = 0;
            MPI_Recv(&bound, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

            cout << id << " RECEIVED BOUND: " << bound << endl; 
        }

        cout << "I'm " << id << endl;

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Printing;

    MPI_Finalize();

}