#include <iostream>
#include <map>
#include <mpi.h>
#include <vector>

#define N 1<<10
#define entryNode 0
#define keyNode (N)-1

using namespace std;


// NOTES
/*
- it is possible to parallelize the cicles with OpenMP? (if not, using CUDA)
- nodeQueue insert at the start and remove from the end (FIFO)
*/


void initializeMatrix(int (&matrix)[N][N], int n){

    for(int i = 0; i<n; i++){
        for(int j = 0; j<n; j++){
            matrix[i][j] = 1;
        }
    }
}

bool Explore(int (&graph)[N][N], int selectedNode, vector<int>& nqueue, map<int, bool>& dmap){

    for(int i = 0; i<N; i++){

        if(graph[selectedNode][i]==1 && dmap.find(i) == dmap.end()){
            if(i == keyNode){
                return true;
            }
            nqueue.push_back(i);
        }
    }
    return false;
}


int main(int argc, char *argv[]){
    

    MPI_Init(&argc, &argv);

    int procsCount;
    int procId;
    MPI_Comm_size(MPI_COMM_WORLD, &procsCount);
    MPI_Comm_rank(MPI_COMM_WORLD, &procId);

    int graph[N][N];
    initializeMatrix(graph, N);

    int nextNode;
    bool sigfound = false;
    vector<int> nodeBuff;
    map<int, bool> doneMap;

    if(procId == 0){

        int elementsCount = 0;

        // If the insertion is O(N) is important to initialize that at the beginning (lets see further)
        doneMap.insert({entryNode,true});
        sigfound = Explore(graph, entryNode, nodeBuff, doneMap);

        while (!sigfound) {
            //NOTES
            /*
            - I don't know if nodeBuff.data() for receiving works well, I have some doubts
            - MPI_Gather preserve original content [:)]
            - MPI_Gather elements count is per process, so this implementation can't work
            */
            if(/*is not the first time*/){
                MPI_Reduce(nullptr, &elementsCount, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
                MPI_Gather(nullptr, 0, MPI_INT, nodeBuff.data(), elementsCount, MPI_INT, 0, MPI_COMM_WORLD);
            
                // Remove repetitions in buff
                // todo
            
                // Remove visited nodes
                vector<int> auxNode;
                for (int i = 0; i<nodeBuff.size(); i++){
                    if(doneMap.find(nodeBuff[i]) == doneMap.end()){
                        auxNode.push_back(nodeBuff[i]);
                    }
                }
                nodeBuff = auxNode;
            }

            // Update doneMap
            for (int i = 0; i<nodeBuff.size(); i++){
                if(doneMap.find(nodeBuff[i]) == doneMap.end()){
                    doneMap.insert({nodeBuff[i], true});
                }
            }

            // NOTES
            /*
            - we need to select better the destination processes 
            - the nodeBuff is reverse iterated to allow the pop_back() usage, but doing that it becomes LIFO, we need a FIFO implementation
            */
            int batchcount = procsCount;
            if(nodeBuff.size()<procsCount){
                batchcount = nodeBuff.size();
            }
            for(int i = batchcount-1; i > 0; i--){
                MPI_Send(&nodeBuff[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                nodeBuff.pop_back();
            }
        }   

    } else {

        // Node receiveing (recv)
        // Analyses of the received node to find outcome nodes
        // Send back the outcome nodes

    }
    




    
}