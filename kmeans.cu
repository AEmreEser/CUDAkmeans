#include <iostream>
#include <fstream>
#include <sstream>
#include <climits>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <cuda_runtime.h>
using namespace std;

#define MAX_ITER 500
#define BLOCK_SIZE 256  // Block size for CUDA kernel, adjust as needed

double inline dist(double x1, double y1, double x2, double y2) {
  return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);  // sqrt is omitted for performance.
}

void randomCenters(int *&x, int *&y, int n, int k, double *&cx, double *&cy) {
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

__global__ void update_kernel(int *x, int *y, int *c, double *cx, double *cy, int k, int n) {
    // Shared memory for block-wise summation
    __shared__ float sumx[BLOCK_SIZE];
    __shared__ float sumy[BLOCK_SIZE];
    __shared__ int count[BLOCK_SIZE];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < k) {
        sumx[tid] = 0.0;
        sumy[tid] = 0.0;
        count[tid] = 0;
    }
    __syncthreads();
    
    // Each thread will add its corresponding values to the shared memory
    if (tid < n) {
        int cl = c[tid];
        atomicAdd(&sumx[cl], (float)x[tid]);
        atomicAdd(&sumy[cl], (float)y[tid]);
        atomicAdd(&count[cl], 1);
    }

    __syncthreads();
    
    // Perform reduction across threads within each block
    if (threadIdx.x == 0) {
        for (int i = 1; i < blockDim.x; i++) {
            sumx[blockIdx.x] += sumx[i];
            sumy[blockIdx.x] += sumy[i];
            count[blockIdx.x] += count[i];
        }

        // Update the centroids with the block-sum data
        if (blockIdx.x == 0) {
            for (int i = 0; i < k; i++) {
                if (count[i] > 0) {
                    cx[i] = sumx[i] / count[i];
                    cy[i] = sumy[i] / count[i];
                }
            }
        }
    }
}

bool assign(int *&x, int *&y, int *&c, double *&cx, double *&cy, int k, int n) {
  bool end = true;
  int * changed = new int[n]();

  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    int cluster;
    int minDist = INT_MAX;
    // Assign to closest center
    for (int j = 0; j < k; j++) {
      double distance = dist(cx[j], cy[j], x[i], y[i]);      
      if (distance < minDist) {
        minDist = distance;
        cluster = j;
      }
    }
    if (cluster != c[i]) {
      changed[i] = 1;
      c[i] = cluster;
    }
  }

  int sum = 0;
  #pragma omp parallel for reduction(+:sum)
  for (int i = 0; i < n; i++){
    sum += changed[i];
  }

  delete[] changed;
  return (sum == 0); 
}

void init(int *&x, int *&y, int *&c, double *&cx, double *&cy, int k, int n) {
  cx = new double[k];
  cy = new double[k];

  randomCenters(x, y, n, k, cx, cy);

  c = new int[n];
  assign(x, y, c, cx, cy, k, n);
}

int readfile(string fname, int *&x, int *&y) {
  ifstream f;
  f.open(fname.c_str());
  string line;
  getline(f,line);
  int n = atoi(line.c_str());

  x = new int[n];
  y = new int[n];

  int tempx, tempy;
  for(int i = 0; i < n; i++) {
    getline(f,line);
    stringstream ss(line);
    ss >> tempx >> tempy;
    x[i] = tempx;
    y[i] = tempy;
  }
  return n;
}

void kmeans(int *&x, int *&y, int *&c, double *&cx, double *&cy, int k, int n) {
  bool end = false; 
  int iter = 0;
  while(!end && iter != MAX_ITER) {
    // CUDA kernel update
    int blockSize = BLOCK_SIZE;
    int gridSize = (n + blockSize - 1) / blockSize;
    update_kernel<<<gridSize, blockSize>>>(x, y, c, cx, cy, k, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA error in kernel launch: %s\n", cudaGetErrorString(err));
      exit(-1);
    }

    cudaDeviceSynchronize();

    end = assign(x, y, c, cx, cy, k, n);
    iter++;
    if (end) {
      printf("end at iter :%d\n", iter);
    }
  }
  printf("Total %d iterations.\n", iter);
}

void print(int *&x, int *&y, int *&c, double *&cx, double *&cy, int k, int n) {
  for (int i = 0; i < k; i++) {
    printf("**Cluster %d **", i);
    printf("**Center :(%f,%f)\n", cx[i], cy[i]);
    for (int j = 0; j < n; j++) {
      if (c[j] == i)
        printf("(%d,%d) ", x[j], y[j]);
    }
    printf("\n");
  }
}

void usage() {
  printf("./kmeans <filename> <k>\n");
  exit(-1);
}

int main(int argc, char *argv[]) {
  if (argc - 1 != 2) {
    usage();
  }

  string fname = argv[1];
  int k = atoi(argv[2]);

  int *x, *y, *c;
  double *cx, *cy;

  int n = readfile(fname, x, y);

  // Measure time for initialization
  double init_start = omp_get_wtime();
  init(x, y, c, cx, cy, k, n);
  double init_end = omp_get_wtime();
  printf("Initialization Time: %f seconds\n", init_end - init_start);

  // Measure time for k-means clustering
  double kmeans_start = omp_get_wtime();
  kmeans(x, y, c, cx, cy, k, n);
  double kmeans_end = omp_get_wtime();
  printf("K-Means Execution Time: %f seconds\n", kmeans_end - kmeans_start);

  // Evaluate clustering quality
  double totalSSD = 0.0;
  for (int i = 0; i < n; i++) {
    int cluster = c[i];
    totalSSD += dist(x[i], y[i], cx[cluster], cy[cluster]);
  }
  printf("Sqrt of Sum of Squared Distances (SSD): %f\n", sqrt(totalSSD));

  // Uncomment to print results
  #ifdef PRINT
  print(x, y, c, cx, cy, k, n);
  #endif

  delete[] x;
  delete[] y;
  delete[] cx;
  delete[] cy;
  delete[] c;

  return 0;
}
