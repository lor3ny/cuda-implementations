#include <algorithm>
#include <cmath>
#include <mpi.h>
#include <ostream>
#include <random>
#include <iostream>
#include <set>

#define N 1<<15


using namespace std;

struct Coords {
    int x;
    int y;
    char location;

    bool operator<(const Coords& other) const {
        if(x == other.x && y == other.y && location == other.location){
            return true;
        }
        return false;
    }
};


int main(int argc, char *argv[]){
    
    MPI_Init(&argc, &argv);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int id;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    int squarel = 2; //so the range is between 0<=x<=2 and 0<=y<=2
    int circler = 1; //so the range is ?

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

        // Non stiamo facendo checking delle ripetizioni, e' necessario
        vector<Coords> s_coords;
        vector<Coords> c_coords; 
        
        int n = N;
        int batchSize = (N)/(size-1);

        // Con Iprobe e' possibile rendere questo processo lavoratore

        MPI_Status status;
        while(s_coords.size()< n){
            int counter = 0;

            float batchSize =  ((float)(N)-s_coords.size()/(float)(size));
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
                    } else if(s_coords.size() < n) {
                        s_coords.push_back(ret);
                        if(ret.location == 'C')
                            c_coords.push_back(ret);
                    }
                    flag = !flag;
                }
                // HANDLE COMUNICATION

                // DO COMPUTATION
                if(s_coords.size() == n){
                    break;
                }
                Coords rand_coord;
                rand_coord.x = (uniform_int_distribution<int>(0,2000)(gen));
                rand_coord.y = (uniform_int_distribution<int>(0,2000)(gen));

                if(sqrt(pow((float)rand_coord.x, 2) + pow(rand_coord.y, 2)) < 1000){
                    rand_coord.location = 'C';
                    c_coords.push_back(rand_coord);
                } else {
                    rand_coord.location = 'S';
                }
                s_coords.push_back(rand_coord);
                
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

        cout << "TOTAL POINTS: " << s_coords.size() << endl;
        cout << "APPROXIMATED PI: " << (float)s_coords.size()/(float)c_coords.size() << endl;
        
    } else {

        random_device rd;
        mt19937 gen(rd());

        int batchN;

        MPI_Bcast(&batchN, 1, MPI_INT, 0, MPI_COMM_WORLD);

        do{
            int counter = 0;
            while(counter < batchN){
                Coords rand_coord;
                rand_coord.x = (uniform_int_distribution<int>(0,2000)(gen));
                rand_coord.y = (uniform_int_distribution<int>(0,2000)(gen));
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