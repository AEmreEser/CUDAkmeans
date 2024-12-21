#include <iostream>
#include <fstream>
#include <sstream>
#include <climits>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <cuda_runtime.h>

#include "reduce.cuh"

using namespace std;

#define MAX_ITER 500
#define BLOCK_SIZE 256  // Block size for CUDA kernel, adjust as needed

// TODO: Pack this in memory
struct Point {
    double x;
    double y;
    Point() : x(0.0), y(0.0) {}
    Point(double x, double y) : x(x), y(y) {}
    Point(Point p) : x(p.x), y(p.y) {}
};

__host__ __device__ double dist(double x1, double y1, double x2, double y2) {
  return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);  // sqrt is omitted for performance.
}

__host__ __device__ double dist(Point a, Point b) {
  return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);  // sqrt is omitted for performance.
}


__host__ void randomCenters(Point*d_points, int n, int k, Point*d_cl_points) {
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
            d_cl_points[added] = d_points[temp];
            centroids[added++] = temp;
        }
    }

    delete[] centroids;
}

// need a way of signaling changes in cluster assignments!!!
// @param bool& change: return value -- true if changed
//  todo: use a global array to keep change info!!! - so that we can do reduction on it afterwards
__global__ void assign_kernel(Point *d_points, int*d_c, Point *d_cl_points, int*d_c_size, bool& change,int n, int k)
{
    // todo: use shared memory
    // __shared__ Point sh_points[BLOCK_SIZE];
    // extern __shared__ Point sh_cl_pt_cache[/* size = k - dynamic */]; // must dynamically allocate

    change = false;
    const int gl_idx = blockIdx.x*blockDim.x + threadIdx.x;
    int minDist = INT_MAX;
    int cluster;
    // if (gl_idx < n){
    //     sh_points[threadIdx.x] = d_points[gl_idx];
        
    // }

    if (gl_idx < n){
        // todo-idea: use k*n threads in total, parallelize the loop below - do min reduction on the cluster value
        for (int cl = 0; cl < k; cl++){
            double dist = dist(d_points[gl_idx], d_cl_points[cl]);
            if (dist < minDist){
                minDist = dist;
                cluster = cl; 
            }
        }
        // divergent branch -- do the stanford bithacks thing
        if (d_c[gl_idx] != cluster){ // cl changed?
            change = true;
            d_c[gl_idx] = cluster;
        }
    }
}

/** PARAMS:
    - d_points -- all n points, global mem
    - d_cl_points -- all k cluster centers, global mem
    - d_c -- all n cluster assignments, global mem
    - d_cx, d_cy, d_count -- k elt long global memory arrays to keep sums and counts for the next cluster calculations -- keeps global sums for each of the k clusters -- global mem

    IMPORTANT: d_cx, d_cy, d_count must be initalized to all zeros before running this for the first time!!!
 */
__device__ void update(Point *&d_points, int *&d_c, Point *&d_cl_points, double*&d_cx, double*&d_cy, int*&d_count, int k, int n) {
    const int gl_idx = blockIdx.x * blockDim.x + threadIdx;
    const int sh_idx = threadIdx;

    // todo: make sure that these fit inside shared memory!!!
    __shared__ double s_points[blockDim.x];// threads per block!!
    __shared__ int    s_c[blockDim.x];
    // copy data to shared memory
    if (gl_idx < n){
        s_points[sh_idx] = d_points[gl_idx];
        s_c[sh_idx] = d_c[gl_idx];
    }

    // must do this before the first run - fence post problem : )
    // if (gl_idx == 0){
    //     for (int i = 0; i < k; i++){
    //         d_cx[i] = 0.0;
    //         d_cy[i] = 0.0;
    //         d_count[i] =0;
    //     }
    // } // absolutely terrible, everybody waits for this guy...
    __syncthreads();

    if (sh_idx == 0){ // one thread per block
        // block local - only used by 1 thread per block
        double *sumx = new double[k]; double sumy* = new double[k];
        int *count = new int[k];

        for (int i = 0; i < k; i++){ // initialize local sum arrays
            sumx[i] = 0.0;
            sumy[i] = 0.0;
            count[i] = 0;
        }

        // naive reduction
        for (int i = 0; i < blockDim.x; i++){ // add block points to the local sum
            int clid = s_c[i];
            sumx[clid] += s_points[i].x;
            sumy[clid] += s_points[i].y;
            count[clid] += 1;
        }

        // cx and cy must be zero entirely!!!
        for (int i =0; i < k; i++){ // add block sum to the global sum
            int clid = s_c[i];
            atomicAdd(&d_cx[i], sumx[clid]);
            atomicAdd(&d_cy[i], sumy[clid]);
            atomicAdd(&d_count[i], count[clid]);
        }

    }
    __syncthreads();

    if (gl_idx < k){ // the first block only, one thread per cluster
        d_cl_points[gl_idx].x = d_cx[gl_idx] / d_count[gl_idx];
        d_cl_points[gl_idx].y = d_cy[gl_idx] / d_count[gl_idx];
        // restore for the next run
        d_cx[gl_idx] = 0.0;
        d_cy[gl_idx] = 0.0;
        d_count[gl_idx] =0;
    }
}


__global__ void kmeans()


__host__ void init(Point*&d_points, int *&c, Point*&d_cl_points, double *&d_cx, double *&d_cy, int*&d_count int k, int n) {
    //   cx = new double[k];
    //   cy = new double[k];
    d_cl_points = new Point[k];
    d_cx = new double[k];
    d_cy = new double[k];
    d_count = new int[k];

    randomCenters(d_points, n, k, d_cl_points);

    c = new int[n];
    // assign(d_points, c, d_cl_points, k, n); // WRONG PAREMETER ORDER (NOTE FOR WHEN YOU REUSE THIS)!!
}

__host__ int readfile(string fname, Point*&d_points) {
    ifstream f;
    f.open(fname.c_str());
    string line;
    getline(f,line);
    int n = atoi(line.c_str());

    // x = new int[n];
    // y = new int[n];
    d_points = new Point[n];

    int tempx, tempy;
    for(int i = 0; i < n; i++) {
        getline(f,line);
        stringstream ss(line);
        ss >> tempx >> tempy;
        d_points[i] = Point(tempx,tempy);
        // x[i] = tempx;
        // y[i] = tempy;
    }
    return n;
}

__host__ void print(Point* d_points, int *c, Point* d_cl_points, int k, int n) {
  for (int i = 0; i < k; i++) {
    printf("**Cluster %d **", i);
    printf("**Center :(%f,%f)\n", d_cl_points[i]);
    for (int j = 0; j < n; j++) {
      if (c[j] == i)
        printf("(%d,%d) ", d_points[j].x, d_points[j].y);
    }
    printf("\n");
  }
}

__host__ void usage() {
    printf("./kmeans <filename> <k>\n");
    exit(-1);
}

int main(int argc, char *argv[]) {
    if (argc - 1 != 2) {
        usage();
    }

    string fname = argv[1];
    int k = atoi(argv[2]);

    //   int *x, *y, *c;
    //   double *cx, *cy;
    Point *d_points, *d_cl_points;

    // int n = readfile(fname, x, y);
    int n = readfile(fname, d_points);

    // Measure time for initialization
    double init_start = omp_get_wtime();
    // init(x, y, c, cx, cy, k, n);
    init(d_points, c, d_cl_points, k, n);
    double init_end = omp_get_wtime();
    printf("Initialization Time: %f seconds\n", init_end - init_start);

    // Measure time for k-means clustering
    double kmeans_start = omp_get_wtime();
    // kmeans(x, y, c, cx, cy, k, n);
    KERNEL LAUNCH HERE!!!!
    kmeans(d_points, c, d_cl_points, k, n);
    double kmeans_end = omp_get_wtime();
    printf("K-Means Execution Time: %f seconds\n", kmeans_end - kmeans_start);

    // Evaluate clustering quality
    double totalSSD = 0.0;
    for (int i = 0; i < n; i++) {
        int cluster = c[i];
        totalSSD += dist(d_points[i], d_cl_points[i]);
    }
    printf("Sqrt of Sum of Squared Distances (SSD): %f\n", sqrt(totalSSD));

    // Uncomment to print results
    #ifdef PRINT
    print(d_points, c, d_cl_points, k, n);
    #endif

    // delete[] x;
    // delete[] y;
    // delete[] cx;
    // delete[] cy;
    delete[] d_points;
    delete[] d_cl_points;
    delete[] c;

    return 0;
}
