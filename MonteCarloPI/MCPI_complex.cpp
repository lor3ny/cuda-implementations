#include <algorithm>
#include <cmath>
#include <mpi.h>
#include <ostream>
#include <random>
#include <iostream>
#include <unordered_set>

#define N 1<<18

// on laptop from 19 included crashes, let's see on pc and cluster

using namespace std;

struct Coords {
    int x;
    int y;
    char location;
};

int main(int argc, char *argv[]){
    
    MPI_Init(&argc, &argv);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int id;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    // DATATYPE CREATION

    MPI_Datatype mpi_coords;

    int blockLengths[3] = {1, 1, 1};
    MPI_Aint displacements[3] = {0, sizeof(int), 2*sizeof(int)};

    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_CHAR};

    MPI_Type_create_struct(3, blockLengths, displacements, types, &mpi_coords);

    MPI_Type_commit(&mpi_coords);
    

    if(id==0){
        random_device rd;
        mt19937 gen(rd());
        
        int n = N;
        int batchSize = (N)/(size);

        // Con Iprobe e' possibile rendere questo processo lavoratore

        MPI_Status status;
        long long s_points = 0;
        long long c_points = 0;
        while(s_points < n){
            int counter = 0;

            float batchSize =  ((float)(N)-s_points/(float)(size));
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
                    Coords ret;
                    MPI_Recv(&ret, 1, mpi_coords, probStatus.MPI_SOURCE, probStatus.MPI_TAG, MPI_COMM_WORLD, &status);
                    if(ret.x == -1){
                        counter++;
                    } else if(s_points < n) {
                        s_points++;
                        if(ret.location == 'C')
                            c_points++;
                        
                    }
                    flag = !flag;
                }
                // HANDLE COMUNICATION

                // DO COMPUTATION
                if(s_points == n){
                    break;
                }
                Coords rand_coord;
                rand_coord.x = (uniform_int_distribution<int>(-1000,1000)(gen));
                rand_coord.y = (uniform_int_distribution<int>(-1000,1000)(gen));

                
                if(sqrt(pow(rand_coord.x, 2) + pow(rand_coord.y, 2)) < 1000){
                    c_points++;
                }
                s_points++;
                
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

        MPI_Barrier(MPI_COMM_WORLD);

        cout << "TOTAL POINTS: " << s_points << endl;
        cout << "CIRCLE POINTS: " << c_points << endl;
        cout << "APPROXIMATED PI: " << 4*((float)c_points/(float)s_points) << endl;
        
    } else {

        random_device rd;
        mt19937 gen(rd());

        int batchN;

        MPI_Bcast(&batchN, 1, MPI_INT, 0, MPI_COMM_WORLD);

        do{
            int counter = 0;
            while(counter < batchN){
                Coords rand_coord;
                rand_coord.x = (uniform_int_distribution<int>(-1000,1000)(gen));
                rand_coord.y = (uniform_int_distribution<int>(-1000,1000)(gen));
                if( sqrt(pow(rand_coord.x, 2) + pow(rand_coord.y, 2)) < 1000){
                    rand_coord.location = 'C';
                } else {
                    rand_coord.location = 'S';
                }
                MPI_Send(&rand_coord, 1, mpi_coords, 0, 0, MPI_COMM_WORLD);
                counter++;
            }
            Coords end_coord;
            end_coord.x = -1;
            MPI_Send(&end_coord, 1, mpi_coords, 0, 0, MPI_COMM_WORLD);

            MPI_Bcast(&batchN, 1, MPI_INT, 0, MPI_COMM_WORLD);
        } while (batchN > 0);

        cout << "I've done " << id << endl;

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Printing;

    MPI_Type_free(&mpi_coords);

    MPI_Finalize();

}