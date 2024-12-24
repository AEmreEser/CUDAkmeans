#include <iostream>
#include <fstream>
#include <sstream>
#include <climits>      /* INT_MAX */
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>
#include <omp.h>

using namespace std;

#ifndef MAX_ITER
#define MAX_ITER 500
#endif

#define CHECK_CUDA_ERROR() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(-1); \
    } \
}

/** TODO:
--1 do the convergence check thing
--2 try to use shared memory as much as you can
--3 Stanford bithacks
**/

// #define DEBUG

__host__ __device__ double dist(double x1, double y1, double x2, double y2) {
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);  // sqrt is omitted as it reduces performance.
}

__global__ void assignKernel(int *x, int *y, int *c, double *cx, double *cy, /* bool *changed, */ int k, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("%d\n", idx);
    if (idx < n) {
        int cluster;
        double minDist = static_cast<double>(INT_MAX);
        // Assign to closest center
        for (int j = 0; j < k; j++) {
            double distance = dist(cx[j], cy[j], x[idx], y[idx]);
            if (distance < minDist) {
                minDist = distance;
                cluster = j;
            }
            // minDist = distance * static_cast<int>(distance < minDist) + minDist * static_cast<int>(!(distance < minDist));
            // cluster = j * static_cast<int>(distance < minDist) + cluster * static_cast<int>(!(distance < minDist));
        }

        // changed[idx] = (c[idx] != cluster);
        // printf("idx=%d, c[idx]=%d, cluster=%d, changed[idx]=%d\n", idx, c[idx], cluster, changed[idx]);
        c[idx] = cluster; // assign the point to the cluster with minDist
        // __threadfence();

    }
}

__global__ void updateKernel(int *x, int *y, int *c, int k, int n, double *sumx, double *sumy, /* bool* changed, bool* cont, */ int *count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&sumx[c[idx]], x[idx]);
        atomicAdd(&sumy[c[idx]], y[idx]);
        atomicAdd(&count[c[idx]], 1);
        // __syncthreads();
    }
    // reduction on changed[]:
    // sub-optimal proof of concept version:
    // __syncthreads();
    // if (idx == 0){
    //     *cont = false;
    //     for (int i = 0; i < n; i++){
    //         *cont |= changed[idx];    
    //         // printf("%d: changed[idx]=%d\n", i, changed[idx]);
    //         if (*cont == true){ break; }
    //     }
    // }
}

__global__ void computeCentroids(int k, double *sumx, double *sumy, int *count, double *cx, double *cy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k) {
        if (count[idx] > 0) {
            cx[idx] = sumx[idx] / count[idx];
            cy[idx] = sumy[idx] / count[idx];
        }
    }
}

void randomCenters(int *x, int *y, int n, int k, double *cx, double *cy) {
  int *centroids = new int[k];

#ifdef RANDOM
  srand (time(NULL)); //normal code
  int added = 0;
  
  while(added != k) {
    bool exists = false;
    int temp = rand() % n;
    for(int i = 0; i < added; i++) {
      if(centroids[i] == temp) {
        exists = true;
      }
    }
    if(!exists) {
      cx[added] = x[temp];
      cy[added] = y[temp];
      centroids[added++] = temp;
    }
  }
#else //deterministic init
  for(int i = 0; i < k; i++) {
     cx[i] = x[i];
     cy[i] = y[i];
     centroids[i] = i;
  }
#endif
delete[] centroids;
}

void kmeans(int *x, int *y, int *c, double *cx, double *cy, int k, int n) {
    bool end = false;
    int iter = 0;
    int * count = new int[k];

    int *d_x, *d_y, *d_c;
    double *d_cx, *d_cy, *d_sumx, *d_sumy;
    int *d_count;
    // bool *d_changed;
    // bool *d_cont;
    // bool *cont;
    // cont = new bool;
    bool cont;
    double *prev_cx, *prev_cy;
    prev_cx = (double*)malloc(k*sizeof(double));
    prev_cy = (double*)malloc(k*sizeof(double));

    cudaMalloc(&d_x, n * sizeof(int));
    cudaMalloc(&d_y, n * sizeof(int));
    cudaMalloc(&d_c, n * sizeof(int));
    cudaMalloc(&d_cx, k * sizeof(double));
    cudaMalloc(&d_cy, k * sizeof(double));
    cudaMalloc(&d_sumx, k * sizeof(double));
    cudaMalloc(&d_sumy, k * sizeof(double));
    cudaMalloc(&d_count, k * sizeof(int));
    // cudaMalloc(&d_changed, n * sizeof(bool));
    // cudaMalloc(&d_cont, 1 * sizeof(bool));
    
    cudaMemcpy(d_x, x, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cx, cx, k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cy, cy, k * sizeof(double), cudaMemcpyHostToDevice);
    printf("begin\n");

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    // cudaMemset(d_c, -1, n * sizeof(int)); // set to some invalid cluster so that every pt gets a cluster change in the first iter

    while (iter < MAX_ITER) {
        #ifdef DEBUG
            printf("iter %d\n", iter);
        #endif
        cont = false;
        
        cudaMemset(d_sumx, 0, k * sizeof(double));
        cudaMemset(d_sumy, 0, k * sizeof(double));
        cudaMemset(d_count, 0, k * sizeof(int));

        assignKernel<<<gridSize, blockSize>>>(d_x, d_y, d_c, d_cx, d_cy, /*d_changed,*/ k, n);
        // cudaDeviceSynchronize();
        
        updateKernel<<<gridSize, blockSize>>>(d_x, d_y, d_c, k, n, d_sumx, d_sumy, /* d_changed, d_cont,*/ d_count);
        cudaDeviceSynchronize(); // wait for gpu
        // cudaMemcpy(cont, d_cont, 1 * sizeof(bool), cudaMemcpyDeviceToHost);
        
        computeCentroids<<<(k + blockSize - 1) / blockSize, blockSize>>>(k, d_sumx, d_sumy, d_count, d_cx, d_cy);
        // cudaDeviceSynchronize();

        cudaMemcpy(cx, d_cx, k * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(cy, d_cy, k * sizeof(double), cudaMemcpyDeviceToHost);
        // cudaDeviceSynchronize(); // there is one sync in main already
        #pragma omp parallel for reduction(|:cont)
        for(int i = 0; i < k; i++){
            cont |= ((cx[i] != prev_cx[i]) || (cy[i] != prev_cy[i]));
        }

        #pragma omp parallel for
        for (int i = 0; i < k; i++){
            prev_cx[i] = cx[i];
            prev_cy[i] = cy[i];
        }

        #ifdef DEBUG
            cudaMemcpy(count, d_count, k * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
            printf("iter %d result\n",iter);
            for (int i = 0; i < k; i++){
                cout << "cluster " << i << " " << cx[i] << ", " << cy[i] << ", count: " << count[i] << endl;
            }
        #endif
        if (cont == false){ 
            printf("iter %d, cont=%d\n", iter, cont);
            break; 
        } // means no changes -- converged

        iter++;
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_c);
    cudaFree(d_cx);
    cudaFree(d_cy);
    cudaFree(d_sumx);
    cudaFree(d_sumy);
    cudaFree(d_count);
    // cudaFree(d_changed);
    // cudaFree(d_cont);
}

int readfile(const string& fname, int*& x, int*& y) {
    ifstream f;
    f.open(fname.c_str());
    if (!f.is_open()) {
        cerr << "Error opening file: " << fname << endl;
        exit(-1);
    }

    string line;
    getline(f, line);  // Read the first line (number of points)
    int n = atoi(line.c_str());
    #ifdef DEBUG
        printf("file length: %d\n", n);
    #endif

    #ifdef ALIGNED_ALLOC
        x = (int*) aligned_alloc(64, n * sizeof(int));
        y = (int*) aligned_alloc(64, n * sizeof(int));
    #else
        x = new int[n];
        y = new int[n];
    #endif

    // Read the points
    int tempx, tempy;
    for (int i = 0; i < n; i++) {
        getline(f, line);
        stringstream ss(line);
        ss >> tempx >> tempy;
        x[i] = tempx;
        y[i] = tempy;
    }

    return n;
}


int main(int argc, char *argv[]) {
    // Check arguments
    if (argc - 1 != 2) {
        printf("./test <filename> <k>\n");
        exit(-1);
    }

    string fname = argv[1];
    int k = atoi(argv[2]);
    
    int *x, *y, *c;
    double *cx, *cy;

    // Read input data
    int n = readfile(fname, x, y);
    
    cx = new double[k];
    cy = new double[k];
    c = new int[n];

    // Initialize centroids
    randomCenters(x, y, n, k, cx, cy);

    // Measure k-means execution time
    double kmeans_start = omp_get_wtime();
    kmeans(x, y, c, cx, cy, k, n);

    cudaDeviceSynchronize();

    double kmeans_end = omp_get_wtime();
    printf("K-Means Execution Time: %f seconds\n", kmeans_end - kmeans_start);

    #ifdef DEBUG
        for (int i = 0; i < k; i++){
            cout << "cluster " << i << " " << cx[i] << ", " << cy[i] << endl;
        }
    #endif

    double totalSSD = 0.0;
    for (int i = 0; i < n; i++) {
        int cluster = c[i];
        totalSSD += dist(x[i], y[i], cx[cluster], cy[cluster]);
    }

    printf("Sqrt of Sum of Squared Distances (SSD): %f\n", sqrt(totalSSD));

    delete[] x;
    delete[] y;
    delete[] cx;
    delete[] cy;
    delete[] c;

    return 0;
}
