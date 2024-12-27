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

#ifndef OUTFILE
#define OUTFILE "out.txt"
#endif

#define CHECK_CUDA_ERROR() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(-1); \
    } \
}

/** TODO:
done: --1 do the convergence check thing 
failed: --2 use n*k threads and parallelize the for loop in assignKernel -- for large k values this is the bottleneck currently
--3 Join the kernels into one big kernel and do everything on the gpu, don't lose time communicating
--4 Use the cuda timer stuff for measuring walltimes - not the omp walltime thing
done:-.5 use Harris reduction on convergence check
--6 comment cleanup
--7 two gpus
**/

// #define DEBUG

__host__ __device__ double dist(double x1, double y1, double x2, double y2) {
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);  // sqrt is omitted as it reduces performance.
}

__global__ void assignKernel(int *x, int *y, int *c, double *cx, double *cy, /* bool *changed, */ int k, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int cluster;
        double minDist = (INT_MAX); // static_cast<double>(INT_MAX); // not using casting saves a miniscule amount of time
        // for really large values of k, the for loop creates a huge bottleneck
        for (int j = 0; j < k; j++) {
            double distance = dist(cx[j], cy[j], x[idx], y[idx]);
            if (distance < minDist) {
                minDist = distance;
                cluster = j;
            }
            // these take more time than the if check above - probably because of the conversion and the multiplication
            // cluster = (j * (distance < minDist)) + (cluster * (!(distance < minDist)));
            // minDist = distance * (distance < minDist) + minDist * (!(distance < minDist));
        }

        // changed[idx] = (c[idx] != cluster);
        // printf("idx=%d, c[idx]=%d, cluster=%d, changed[idx]=%d\n", idx, c[idx], cluster, changed[idx]);
        c[idx] = cluster; // assign the point to the cluster with minDist
        // __threadfence();

    }
}

// __global__ void assignKernelFullyParallel(int *x, int *y, int *c, double *cx, double *cy, double* all_dists, int* all_assignments, int k, int n) {
//     int idx = blockDim.x * blockIdx.x + threadIdx.x;
//     int maxthreads = n*k;
//     if(idx >= maxthreads){ return; }

//     int kidx = threadIdx.x; int nidx = blockIdx.x;

//     // extern __shared__ double dists_to_centroids[]; // size == k*sizeof(double) + k*sizeof(long long int)

//     // dists_to_centroids[kidx] = dist(x[nidx], y[nidx], cx[kidx], cy[kidx]); 
//     // dists_to_centroids[k + kidx] = kidx;
//     all_dists[idx] = dist(x[nidx], y[nidx], cx[kidx], cy[kidx]); 
//     all_assignments[idx] = -1;

//     __syncthreads();

//     for(long long int s=1; s < k; s*=2){ // assume k is divisible by 2
//         if(kidx % (2*s) == 0){
//             // if(dists_to_centroids[kidx] > dists_to_centroids[kidx + s]){
//             //     dists_to_centroids[kidx] = dists_to_centroids[kidx + s]; // min reduction
//             //     dists_to_centroids[k + kidx] = __longlong_as_double(s);  // total goddamn waste
//             // } // if
//             printf("idx=%d, all_dists[idx]=%f, all_dists[idx+s]=%f\n", idx ,all_dists[idx], all_dists[idx+s]);
//             if (all_dists[idx] > all_dists[idx+s]){
//                 all_dists[idx] = all_dists[idx+s];
//                 all_assignments[idx] = static_cast<int>(s);
//             }
//             printf("nidx=%d, kidx=%d, dists=%f, kidx+s=%d, dists=%f\n", nidx, kidx, all_dists[kidx], kidx+s, all_dists[kidx+s]);
//         } // if %
//         __syncthreads();
//     } // for

//     if (kidx == 0){
//         c[nidx] = (all_assignments[idx]);
//         printf("nidx=%d, c[nidx]=%d\n", nidx, c[nidx]);
//     }

// }

// single block, k many threads
__global__ void check_cluster_change(double *cx, double *cy, double *prev_cx, double *prev_cy, bool* changed, bool *red_change, int k) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadId > k){ return; }

    if (threadId == 0){ *red_change = false; }

    // bool local_cont = false;

    // Each thread checks one element
    // if (threadId < k) {
        // local_cont = (cx[threadId] != prev_cx[threadId]) || (cy[threadId] != prev_cy[threadId]);
    // not that big a reduction in perf since global mem accesses are coalesced
    changed[threadId] = (cx[threadId] != prev_cx[threadId]) || (cy[threadId] != prev_cy[threadId]);
    // }

    // Store the local result in shared memory
    // shared_cont[threadIdx.x] = local_cont;
    __syncthreads();

    // Perform parallel reduction in shared memory using OR operation
    // FOR NOW ASSUME THAT K IS AN INTEGER MULTIPLE OF 2
    for (int s = 1; s < k; s *= 2) { 
        int index = 2 * s * threadId;
        // if (threadId % (2 * s) == 0) { // highly divergent
        if(index < k){ // not divergent 
            changed[threadId] |= changed[threadId + s];
        }
        __syncthreads();
    }
    // Proved suboptimal
    // for (int s = k/2; s>0; s>>=1){
    //     if (threadId < s){
    //         changed[threadId] |= changed[threadId + s];
    //     }
    //     __syncthreads();
    // }

    prev_cx[threadId] = cx[threadId];
    prev_cy[threadId] = cy[threadId];

    // The first thread in each block writes the result to global memory
    if (threadId == 0) {
        // atomicOr(cont, changed[0]);
        *red_change |= changed[0];
        // printf("Post Reduction *red_change=%d\n", *red_change);
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
        // for some reason these affect performance badly, probably due to the multiplications
        // cx[idx] = (sumx[idx] / count[idx]) * (count[idx] > 0) + cx[idx] * (count[idx] <= 0);
        // cy[idx] = (sumy[idx] / count[idx]) * (count[idx] > 0) + cy[idx] * (count[idx] <= 0);
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

void writeClusterAssignments(const int* x, const int* y, const int* c, int n, const string& filename) {
    ofstream outFile(filename);
    if (!outFile) {
        throw runtime_error("Could not open file: " + filename);
    }
    
    for (int i = 0; i < n; i++) {
        outFile << x[i] << " " << y[i] << " " << c[i] << "\n";
    }
    
    outFile.close();
}

void kmeans(int *x, int *y, int *c, double *cx, double *cy, int k, int n) {
    cudaEvent_t start, stop, overall_start, overall_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&overall_start);
    cudaEventCreate(&overall_stop);
    cudaEventRecord(overall_start, 0);

    int iter = 0;
    int * count = new int[k];
    double acc_red_time = 0.0f, acc_assign_time = 0.0f, acc_update_time = 0.0f, acc_centroid_time = 0.0f;

    int *d_x, *d_y, *d_c;
    double *d_cx, *d_cy, *d_sumx, *d_sumy; 
    // double *d_all_dists;
    // int *d_all_assignments;
    int *d_count;
    // bool *cont;
    // cont = new bool;
    bool cont;
    #ifdef HOSTREDUCE
        double *prev_cx, *prev_cy;
        #ifdef PINMEM
            cudaMallocHost(&prev_cx, k*sizeof(double));
            cudaMallocHost(&prev_cy, k*sizeof(double));
        #else
            prev_cx = (double*)malloc(k*sizeof(double));
            prev_cy = (double*)malloc(k*sizeof(double));
        #endif
    #else
        double *d_prev_cx, *d_prev_cy;
        bool *d_changed;
        bool *d_red_change;
        cudaMalloc(&d_red_change, 1 * sizeof(bool));
        cudaMalloc(&d_prev_cx, k * sizeof(double));
        cudaMalloc(&d_prev_cy, k * sizeof(double));
        cudaMalloc(&d_changed, k * sizeof(bool));

        // cudaMemset(d_prev_cx, -1.0f, k*sizeof(double));
        // cudaMemset(d_prev_cy, -1.0f, k*sizeof(double));
        cudaMemset(d_changed, false, k*sizeof(bool));
        cudaMemset(d_red_change,    false, 1*sizeof(bool));
    #endif

    cudaMalloc(&d_x, n * sizeof(int));
    cudaMalloc(&d_y, n * sizeof(int));
    cudaMalloc(&d_c, n * sizeof(int));
    cudaMalloc(&d_cx, k * sizeof(double));
    cudaMalloc(&d_cy, k * sizeof(double));
    cudaMalloc(&d_sumx, k * sizeof(double));
    cudaMalloc(&d_sumy, k * sizeof(double));
    cudaMalloc(&d_count, k * sizeof(int));
    // cudaMalloc(&d_all_dists, n*k* sizeof(double));
    // cudaMalloc(&d_all_assignments, n*k* sizeof(int));
    // cudaMalloc(&d_changed, n * sizeof(bool));
    // cudaMalloc(&d_cont, 1 * sizeof(bool));
    
    cudaMemcpy(d_x, x, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cx, cx, k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cy, cy, k * sizeof(double), cudaMemcpyHostToDevice);
    printf("Begin\n");

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    cudaMemset(d_c, -1, n * sizeof(int)); // set to some invalid cluster so that every pt gets a cluster change in the first iter
    // cudaMemset(d_all_dists, -1.0f, n*k* sizeof(double)); // set to some invalid cluster so that every pt gets a cluster change in the first iter
    // cudaMemset(d_all_assignments, -1, n*k* sizeof(int)); // set to some invalid cluster so that every pt gets a cluster change in the first iter

    while (iter < MAX_ITER) {
        #ifdef DEBUG
            printf("==========%d=========\n", iter);
        #endif
        cont = false;
        
        cudaMemset(d_sumx, 0, k * sizeof(double));
        cudaMemset(d_sumy, 0, k * sizeof(double));
        cudaMemset(d_count, 0, k * sizeof(int));

        cudaEventRecord(start, 0);
        assignKernel<<<gridSize, blockSize>>>(d_x, d_y, d_c, d_cx, d_cy, /*d_changed,*/ k, n);
        // assignKernelFullyParallel<<< (n*k + k - 1)/k, k >>>(d_x, d_y, d_c, d_cx, d_cy, d_all_dists, d_all_assignments, /*d_changed,*/ k, n);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float timer_assign;
        cudaEventElapsedTime(&timer_assign, start, stop);
        acc_assign_time += timer_assign;
        
        cudaEventRecord(start, 0);
        updateKernel<<<gridSize, blockSize>>>(d_x, d_y, d_c, k, n, d_sumx, d_sumy, /* d_changed, d_cont,*/ d_count);
        cudaDeviceSynchronize(); // wait for gpu
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float timer_update;
        cudaEventElapsedTime(&timer_update, start, stop);
        acc_update_time += timer_update;
        
        cudaEventRecord(start, 0);
        computeCentroids<<<(k + blockSize - 1) / blockSize, blockSize>>>(k, d_sumx, d_sumy, d_count, d_cx, d_cy);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float timer_centroid;
        cudaEventElapsedTime(&timer_centroid, start, stop);
        acc_centroid_time += timer_centroid;

        #ifdef HOSTREDUCE
            timer_begin = omp_get_wtime();
            cudaMemcpy(cx, d_cx, k * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(cy, d_cy, k * sizeof(double), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize(); // there is one sync in main already
            timer_end = omp_get_wtime();
            // printf("Data migration (device to host): %f\n", timer_end - timer_begin);
            acc_data_transfer_time += timer_end - timer_begin;
            // easiest way to do this is in here

            timer_begin = omp_get_wtime();
            #pragma omp parallel for reduction(|:cont)
            for(int i = 0; i < k; i++){
                cont |= ((cx[i] != prev_cx[i]) || (cy[i] != prev_cy[i]));
            }

            #pragma omp parallel for
            for (int i = 0; i < k; i++){
                prev_cx[i] = cx[i];
                prev_cy[i] = cy[i];
            }
            timer_end = omp_get_wtime();
            // printf("Host Reduction: %f\n", timer_end - timer_begin);
            acc_red_time += timer_end - timer_begin;
        #else
            cudaEventRecord(start, 0);
            check_cluster_change<<<k, 1>>>(d_cx, d_cy, d_prev_cx, d_prev_cy, d_changed, d_red_change, k);
            cudaDeviceSynchronize();
            cudaMemcpy(&cont, d_red_change, 1 *sizeof(bool), cudaMemcpyDeviceToHost);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float timer_reduce;
            cudaEventElapsedTime(&timer_reduce, start, stop);
            acc_red_time += timer_reduce;
        #endif


        #ifdef DEBUG
            #ifndef HOSTREDUCE // NOT defiend
                cudaMemcpy(cx, d_cx, k * sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(cy, d_cy, k * sizeof(double), cudaMemcpyDeviceToHost);
            #endif
            cudaMemcpy(count, d_count, k * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
            printf("End of itertion %d-- results\n",iter);
            for (int i = 0; i < k; i++){
                cout << "cluster " << i << ": (" << cx[i] << ", " << cy[i] << ") , count: " << count[i] << endl;
            }
        #endif

        if (cont == false){ 
            #ifndef HOSTREDUCE // NOT defiend
                cudaMemcpy(cx, d_cx, k * sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(cy, d_cy, k * sizeof(double), cudaMemcpyDeviceToHost);
            #endif
            cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
            printf("Converged at iteration %d\n", iter);
            writeClusterAssignments(x, y, c, n, OUTFILE);
            
            break; 
        } // means no changes -- converged

        iter++;
    }

    #ifdef HOSTREDUCE
        printf("HOST Acc Reduction: %f\n", acc_red_time);
        printf("Acc Data Transfer: %f\n", acc_data_transfer_time);
    #else
        printf("Acc Reduction: %f\n", acc_red_time);
    #endif


    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_c);
    cudaFree(d_cx);
    cudaFree(d_cy);
    cudaFree(d_sumx);
    cudaFree(d_sumy);
    cudaFree(d_count);
    #ifdef PINMEM
        cudaFreeHost(prev_cx);
        cudaFreeHost(prev_cy);
    #endif
    #ifndef HOSTREDUCE // NOT defined
        cudaFree(d_changed);
        cudaFree(d_prev_cx);
        cudaFree(d_prev_cy);
        cudaFree(d_red_change);
    #endif

    cudaEventRecord(overall_stop, 0);
    cudaEventSynchronize(overall_stop);
    float timer_overall;
    cudaEventElapsedTime(&timer_overall, overall_start, overall_stop);
    printf("Kmeans total runtime: %f\n", timer_overall);
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
        printf("./kmeans <filename> <k>\n");
        exit(-1);
    } else if (atoi(argv[2]) < 1){
        printf("k must be a positive integer number\n");
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

    // cudaDeviceSynchronize(); // sync inside kmeans func

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
