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

        // Con Iprobe e' possibile rendere questo processo lavoratore

        MPI_Status status;
        while(unique_numbers.size() < n){
            int counter = 0;

            float batchSize =  ((float)(N)-unique_numbers.size())/(float)(size);
            if(batchSize == 0){
                break;
            }

            int roundedSize = ceil(batchSize);

            MPI_Bcast(&roundedSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

            int flag;
            MPI_Status probStatus;
            int batchCounter;
            do {
                // HANDLE COMUNICATION
                MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &probStatus);

                if(flag){
                    int ret;
                    MPI_Recv(&ret, 1, MPI_FLOAT, probStatus.MPI_SOURCE, probStatus.MPI_TAG, MPI_COMM_WORLD, &status);
                    if(ret == -1){
                        counter++;
                    } else if(unique_numbers.size() < n) {
                        unique_numbers.insert(ret);
                    }
                    flag = !flag;
                }
                // HANDLE COMUNICATION

                // DO COMPUTATION
                if(unique_numbers.size() == n){
                    break;
                }
                float rand_n = ((float) uniform_int_distribution<int>(0,2000)(gen));
                unique_numbers.insert(rand_n);
                batchCounter++;

                if(batchCounter >= roundedSize){
                    counter++;
                }
                // DO COMPUTATION
            
            } while (counter < size);
        }

        batchSize = 0;
        MPI_Bcast(&batchSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
        cout << "I'm " << id << endl;

        cout << unique_numbers.size() << endl;

        MPI_Barrier(MPI_COMM_WORLD);

        // Create and MPI_Datatype to handle not int but tuple of int
        // Sign which are inside the circle and which are outside, with the norm 
        
    } else {

        random_device rd;
        mt19937 gen(rd());

        int batchN;

        MPI_Bcast(&batchN, 1, MPI_INT, 0, MPI_COMM_WORLD);

        do{
            int counter = 0;
            while(counter < batchN){
                float rand_n = ((float) uniform_int_distribution<int>(0,2000)(gen));
                MPI_Send(&rand_n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                counter++;
            }

            int ending = -1;
            MPI_Send(&ending, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

            MPI_Bcast(&batchN, 1, MPI_INT, 0, MPI_COMM_WORLD);
        } while (batchN > 0);

        cout << "I've done " << id << endl;

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Printing;

    MPI_Finalize();

}