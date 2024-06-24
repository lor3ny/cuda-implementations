#include <cstddef>
#include <iostream>
#include <map>
#include <math.h>
#include <mpi.h>
#include <vector>
#include <random>
#include <list>
#include<cstdlib>

#define N 50
#define entryNode 0
#define endNode (N)-1

using namespace std;
// NOTES
/*
- it is possible to parallelize the cicles with OpenMP? (if not, using CUDA)
- nodeQueue insert at the start and remove from the end (FIFO)
*/


void initializeMatrix(int* data){
    srand((unsigned) time(NULL));
    for(int i = 0; i<N; i++){
        for(int j = 0; j<N; j++){
            int random = 0 + (rand() % 10);
            if(random == 0){
                data[i*N + j] = 1;
            } else {
                data[i*N + j] = 0;
            }
            //cout << data[i*N + j];
        }
        //cout << endl;
    }
}



int main(int argc, char *argv[]){
    
    MPI_Init(&argc, &argv);

    int procsCount;
    int procId;
    MPI_Comm_size(MPI_COMM_WORLD, &procsCount);
    MPI_Comm_rank(MPI_COMM_WORLD, &procId);
    int batchCount = (N/procsCount);

    int graph[N*N];
    int myGraph[batchCount*N];

    int startNode;
    bool entryCheck = false;
    bool endCheck = false;
    int sigtermCheck = 0;

    if(procId == 0){

        // SETUP

        //Maybe its better to choose in a more sparse way
        int starters[procsCount];
        for(int i = 0; i<procsCount; i++){
            if(i == 0){
                starters[i] = entryNode;
                continue;
            }
            
            if(i == procsCount-1){
                starters[i] = endNode;
                continue;
            }

            starters[i] = i;
        }
        
        //Send Entry Nodes
        MPI_Scatter(starters, 1, MPI_INT, &startNode, 1, MPI_INT, 0, MPI_COMM_WORLD);

        //Initialize and Scatter matrix
        initializeMatrix(graph);
        MPI_Scatter(graph, batchCount*N, MPI_INT, myGraph, batchCount*N, MPI_INT, 0, MPI_COMM_WORLD);

        // BFS COMPUTATION
        vector<int> path;
        vector<bool> visited;
        list<int> nodeQueue;

        for (int i = 0; i < N; i++)
            visited.push_back(false);
        visited[startNode] = true;
        nodeQueue.push_back(startNode);
        entryCheck = true;


        while (!endCheck) {

            MPI_Status statusHandler;
            int flagHandler;
            MPI_Request requestHandler;
            int vertex = 0;
            MPI_Irecv(&vertex, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &requestHandler);

            MPI_Test(&requestHandler, &flagHandler, &statusHandler);
            if(flagHandler){
                int localVertex = vertex - (procId * batchCount);
                MPI_Request tempReq;
                MPI_Isend(&myGraph[localVertex], N, MPI_INT, statusHandler.MPI_SOURCE, 0, MPI_COMM_WORLD, &tempReq);
            }

            if(nodeQueue.empty()){
                break;
            }

            int currVertex = nodeQueue.front();
            nodeQueue.pop_front();


            int procHandler = floor(currVertex/batchCount);
            //Compute the process that handles the batch
            int* sharedLine;
            MPI_Status tempStatus;
            if(procHandler != procId){
                sharedLine = new int[N];
            
                int err = MPI_Send(&currVertex, 1, MPI_INT, procHandler, 0, MPI_COMM_WORLD);
                MPI_Recv(sharedLine, N, MPI_INT, procHandler, 0, MPI_COMM_WORLD, &tempStatus);
            }

            path.push_back(currVertex);

            for (int i = 0; i< N; ++i) {

                int value = graph[currVertex*N + i];
                if(i == endNode && value == 1){
                    endCheck = true;
                    currVertex = i;
                    path.push_back(currVertex);
                    break;
                }

                if (value == 1 && !visited[i]) {
                    visited[i] = true;
                    nodeQueue.push_back(i);
                }
            }

            delete [] sharedLine;
        }

        cout << procId << " start node: " << startNode << " [end-entry] = [" << endCheck << "-"<< entryCheck << "]" <<  " visited: ";
        for (int i = 0; i < path.size(); i++) {
            cout << path[i] << "-";
        }   
        cout << endl;

        int err;
        MPI_Abort(MPI_COMM_WORLD, err);

    } else {

        // INITIALIZATION

        // Recv entry point
        MPI_Scatter(nullptr, 0, MPI_INT, &startNode, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if(startNode == endNode){
            endCheck = true;
        }

        // Recv the matrix
        //MPI_Bcast(graph, N*N, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatter(nullptr, 1, MPI_INT, myGraph, batchCount*N, MPI_INT, 0, MPI_COMM_WORLD);

        // BFS Computation
        vector<int> path;
        vector<bool> visited;
        list<int> nodeQueue;

        for (int i = 0; i < N; i++)
            visited.push_back(false);
        visited[startNode] = true;
        nodeQueue.push_back(startNode);
        

        int checkPass = -1; 
        
        while (!(endCheck && entryCheck)) {

            MPI_Status statusHandler;
            int flagHandler;
            MPI_Request requestHandler;
            int vertex = 0;
            MPI_Irecv(&vertex, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &requestHandler);

            MPI_Test(&requestHandler, &flagHandler, &statusHandler);
            if(flagHandler){
                int localVertex = vertex - (procId * batchCount);
                MPI_Request tempReq;
                MPI_Isend(&myGraph[localVertex], N, MPI_INT, statusHandler.MPI_SOURCE, 0, MPI_COMM_WORLD, &tempReq);
            }

            if(nodeQueue.empty()){
                break;
            }

            int currVertex;
            if(checkPass != -1){
                currVertex = checkPass;
                checkPass = -1;
            } else {
                currVertex = nodeQueue.front();
                nodeQueue.pop_front();
            }

            int procHandler = floor(currVertex/batchCount);
            int* sharedLine;
            MPI_Status tempStatus;
            if(procHandler != procId){
                sharedLine = new int[N];
                
                /* Verification of the status of a process*/

                MPI_Send(&currVertex, 1, MPI_INT, procHandler, 0, MPI_COMM_WORLD);
                MPI_Recv(sharedLine, N, MPI_INT, procHandler, 0, MPI_COMM_WORLD, &tempStatus);
            }

            path.push_back(currVertex);

            for (int i = 0; i< N; ++i) {

                int value = graph[currVertex*N + i];

                if(i == endNode && value == 1 && !endCheck){
                    if(entryCheck){
                        endCheck = true;
                        path.push_back(i);
                        break;
                    }
                    endCheck = true;
                    visited[i] = true;
                    checkPass = i;
                    continue;
                }

                if(i == entryNode && value == 1 && !entryCheck){
                    if(endCheck){
                        entryCheck = true;
                        path.push_back(i);
                        break;
                    }   
                    entryCheck = true;
                    visited[i] = true;
                    checkPass = i;
                    continue;
                }

                if (value == 1 && !visited[i]) {
                    visited[i] = true;
                    nodeQueue.push_back(i);
                }
            }
        }

        while (nodeQueue.empty());
    
        // PRINT STATUS
        cout << procId << " start node: " << startNode << " [end-entry] = [" << endCheck << "-"<< entryCheck << "]" <<  " visited: ";
        for (int i = 0; i < path.size(); i++) {
            cout << path[i] << "-";
        }   
        cout << endl;

        int err;
        MPI_Abort(MPI_COMM_WORLD, err);

    }   
    


    MPI_Finalize();

    
}