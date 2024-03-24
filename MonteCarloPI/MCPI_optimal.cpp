#include <mpi.h>
#include <iostream>
#include <cmath>
#include <random>

#define N 1<<30

using namespace std;


int main(int argc, char *argv[]){
    
    MPI_Init(&argc, &argv);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int id;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    int c_points = 0;

    if(id==0){
        random_device rd;
        mt19937 gen(rd());
        
        int n = N;
        int batchSize = (N)/(size);
        int s_points = n;

        MPI_Bcast(&batchSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

        for(int i = 0; i<batchSize; i++){
            int x = (uniform_int_distribution<int>(-1000,1000)(gen));
            int y = (uniform_int_distribution<int>(-1000,1000)(gen));
            if( sqrt(pow(x, 2) + pow(y, 2)) < 1000){
                c_points++;
            }
        }
        
        int summedCirclePoints = 0;
        MPI_Reduce(&c_points, &summedCirclePoints, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        cout << "TOTAL POINTS: " << s_points << endl;
        cout << "CIRCLE POINTS: " << c_points << endl;
        cout << "APPROXIMATED PI: " << 4*((float)summedCirclePoints/(float)s_points) << endl;
        
    } else {

        random_device rd;
        mt19937 gen(rd());

        int batchN;

        MPI_Bcast(&batchN, 1, MPI_INT, 0, MPI_COMM_WORLD);

        for(int i = 0; i<batchN; i++){
            int x = (uniform_int_distribution<int>(-1000,1000)(gen));
            int y = (uniform_int_distribution<int>(-1000,1000)(gen));
            if( sqrt(pow(x, 2) + pow(y, 2)) < 1000){
                c_points++;
            }
        }

        MPI_Reduce(&c_points, nullptr, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    }

    MPI_Finalize();

}