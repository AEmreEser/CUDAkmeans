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
#define MAX_ITER 30
#endif

// CUDA kernel to compute the distance between points and centroids
__host__ __device__ double dist(double x1, double y1, double x2, double y2) {
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);  // sqrt is omitted as it reduces performance.
}

// CUDA kernel to assign each point to the nearest centroid
__global__ void assignKernel(int *x, int *y, int *c, double *cx, double *cy, int k, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int cluster;
        int minDist = INT_MAX;
        // Assign to closest center
        for (int j = 0; j < k; j++) {
            double distance = dist(cx[j], cy[j], x[idx], y[idx]);
            if (distance < minDist) {
                minDist = distance;
                cluster = j;
            }
        }
        c[idx] = cluster; // assign the point to the cluster with minDist
    }
}

// CUDA kernel to update the centroids based on the assigned clusters
__global__ void updateKernel(int *x, int *y, int *c, double *cx, double *cy, int k, int n, double *sumx, double *sumy, int *count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&sumx[c[idx]], x[idx]);
        atomicAdd(&sumy[c[idx]], y[idx]);
        atomicAdd(&count[c[idx]], 1);
    }
}

// CUDA kernel to compute the new centroids after update
__global__ void computeCentroids(int k, double *sumx, double *sumy, int *count, double *cx, double *cy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k) {
        if (count[idx] > 0) {
            cx[idx] = sumx[idx] / count[idx];
            cy[idx] = sumy[idx] / count[idx];
        }
    }
}

// Function to initialize centroids randomly
void randomCenters(int *x, int *y, int n, int k, double *cx, double *cy) {
    int *centroids = new int[k];
    srand(time(NULL));
    int added = 0;

    while (added != k) {
        bool exists = false;
        int temp = rand() % n;
        for (int i = 0; i < added; i++) {
            if (centroids[i] == temp) {
                exists = true;
            }
        }
        if (!exists) {
            cx[added] = x[temp];
            cy[added] = y[temp];
            centroids[added++] = temp;
        }
    }

    delete[] centroids;
}

// Main CUDA-based k-means function
void kmeans(int *x, int *y, int *c, double *cx, double *cy, int k, int n) {
    bool end = false;
    int iter = 0;
    int * count = new int[k];

    // Allocate memory on the device
    int *d_x, *d_y, *d_c;
    double *d_cx, *d_cy, *d_sumx, *d_sumy;
    int *d_count;
    
    cudaMalloc(&d_x, n * sizeof(int));
    cudaMalloc(&d_y, n * sizeof(int));
    cudaMalloc(&d_c, n * sizeof(int));
    cudaMalloc(&d_cx, k * sizeof(double));
    cudaMalloc(&d_cy, k * sizeof(double));
    cudaMalloc(&d_sumx, k * sizeof(double));
    cudaMalloc(&d_sumy, k * sizeof(double));
    cudaMalloc(&d_count, k * sizeof(int));
    
    cudaMemcpy(d_x, x, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cx, cx, k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cy, cy, k * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Iterations for K-means
    while (iter < MAX_ITER) {
        printf("iter %d\n", iter);
        // Reset the sumx, sumy, and count arrays on the device
        cudaMemset(d_sumx, 0, k * sizeof(double));
        cudaMemset(d_sumy, 0, k * sizeof(double));
        cudaMemset(d_count, 0, k * sizeof(int));

        // Step 1: Assign points to closest centroids
        assignKernel<<<gridSize, blockSize>>>(d_x, d_y, d_c, d_cx, d_cy, k, n);

        // Step 2: Update centroids
        updateKernel<<<gridSize, blockSize>>>(d_x, d_y, d_c, d_cx, d_cy, k, n, d_sumx, d_sumy, d_count);
        
        // Step 3: Compute new centroids
        computeCentroids<<<(k + blockSize - 1) / blockSize, blockSize>>>(k, d_sumx, d_sumy, d_count, d_cx, d_cy);

        // Copy centroids back to host
        cudaMemcpy(cx, d_cx, k * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(cy, d_cy, k * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(count, d_count, k * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        for (int i = 0; i < k; i++){
            cout << "cluster " << i << " " << cx[i] << ", " << cy[i] << ", count: " << count[i] << endl;
        }

        iter++;
    }

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_c);
    cudaFree(d_cx);
    cudaFree(d_cy);
    cudaFree(d_sumx);
    cudaFree(d_sumy);
    cudaFree(d_count);
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
    printf("file length: %d\n", n);

    // Allocate memory for coordinates
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
    // You can use the readfile function from the original code to load your data
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

    for (int i = 0; i < k; i++){
        cout << "cluster " << i << " " << cx[i] << ", " << cy[i] << endl;
    }

    // Evaluate clustering quality (optional)
    double totalSSD = 0.0;
    // double prev_total = 0.0;
    for (int i = 0; i < n; i++) {
        int cluster = c[i];
        totalSSD += dist(x[i], y[i], cx[cluster], cy[cluster]);
        // assert(totalSSD > prev_total); // no overflow
        // printf("totalSSD: %f\n", totalSSD);
        // prev_total=totalSSD;
    }
    printf("Sqrt of Sum of Squared Distances (SSD): %f\n", sqrt(totalSSD));

    for (int i = 20; i < 28; i++){
        int cluster = c[i];
        cout << i << " assigned cluster: " << c[i] << endl;
        cout << i << " " << "dist( x:"<< x[i] << ", y:" << y[i]<< ", cx:" << cx[cluster] << ", cy:" << cy[cluster] << ")" << endl;
        cout << i << " dist:" << dist(x[i], y[i], cx[cluster], cy[cluster]) << endl;
    }

    delete[] x;
    delete[] y;
    delete[] cx;
    delete[] cy;
    delete[] c;

    return 0;
}
