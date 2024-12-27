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

/** 
    IMPORTANT:
        *RECOMMENDEDCONFIG: 
            FLAT = not defined, HOSTREDUCE = not defined
        *EXPLANATIONS:
            both FLAT and HOSTREDUCE result in significant performance penalties -- they will both be turned off by default
            FLAT produces wrong results for k > 1024 --> the program will automatically use regular assign kernel instead of FLAT if FLAT is specified for k > 1024
 */

#ifndef MAX_ITER
#define MAX_ITER 500
#endif

#ifndef OUTFILE
#define OUTFILE "out_cuda.txt"
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
canceled: --3 Join the kernels into one big kernel and do everything on the gpu, don't lose time communicating
done:--4 Use the cuda timer stuff for measuring walltimes - not the omp walltime thing
done:-.5 use Harris reduction on convergence check
done:--6 comment cleanup
done:--7 two gpus
**/

__host__ __device__ double dist(double x1, double y1, double x2, double y2) {
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);  // sqrt is omitted as it reduces performance.
}

// Helper function for atomic minimum operation on doubles
__device__ void atomicMin_double(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull;
    unsigned long long assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(min(val, __longlong_as_double(assumed))));
    } while (assumed != old);
}

__global__ void assignKernel_flat_old(int *x, int *y, int *c, double *cx, double *cy, int k, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int point_idx = idx / k;  // Which point this thread is working on
    int cluster_idx = idx % k;  // Which cluster this thread is evaluating
    
    if (point_idx < n) {
        // Each thread computes distance for its point-cluster pair
        double distance = dist(cx[cluster_idx], cy[cluster_idx], x[point_idx], y[point_idx]);
        
        // Shared memory to store distances and corresponding clusters
        extern __shared__ double shared_data[];
        double* shared_distances = shared_data;
        int* shared_clusters = (int*)&shared_data[blockDim.x];
        
        // Store this thread's results in shared memory
        shared_distances[threadIdx.x] = distance;
        shared_clusters[threadIdx.x] = cluster_idx;
        __syncthreads();
        
        // First thread for each point finds the minimum among all clusters
        if (cluster_idx == 0) {
            double min_dist = distance;
            int best_cluster = cluster_idx;
            
            // Find minimum distance among all clusters for this point
            int point_start = (threadIdx.x / k) * k;  // Start index for this point's clusters
            for (int j = 1; j < k; j++) {
                int idx_in_block = point_start + j;
                if (idx_in_block < blockDim.x) {
                    if (shared_distances[idx_in_block] < min_dist) {
                        min_dist = shared_distances[idx_in_block];
                        best_cluster = shared_clusters[idx_in_block];
                    }
                }
            }
            
            // Write final result for this point
            c[point_idx] = best_cluster;
        }
    }
}

__global__ void assignKernel_flat(int *x, int *y, int *c, double *cx, double *cy, int k, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int point_idx = idx / k;  // Which point this thread is working on
    int cluster_idx = idx % k;  // Which cluster this thread is evaluating
    
    if (point_idx < n) {
        // Calculate distance for this point-cluster pair
        double distance = dist(cx[cluster_idx], cy[cluster_idx], x[point_idx], y[point_idx]);
        
        // Use shared memory to track minimum distance for each point within a block
        __shared__ struct {
            double dist;
            int cluster;
        } shared_min[256];  // Assume block size <= 256
        
        // Thread's position within its point group in the block
        int local_idx = threadIdx.x % k;
        int point_group_idx = threadIdx.x / k;
        
        // Initialize shared memory
        if (local_idx == 0) {
            shared_min[point_group_idx].dist = INT_MAX;
            shared_min[point_group_idx].cluster = -1;
        }
        __syncthreads();
        
        // Atomically update minimum distance for this point
        atomicMin_double(&shared_min[point_group_idx].dist, distance);
        __syncthreads();
        
        // If this thread found the minimum distance, record its cluster
        if (distance == shared_min[point_group_idx].dist) {
            shared_min[point_group_idx].cluster = cluster_idx;
        }
        __syncthreads();
        
        // Only one thread per point writes the final result
        if (local_idx == 0) {
            c[point_idx] = shared_min[point_group_idx].cluster;
        }
    }
}

__global__ void assignKernel(int *x, int *y, int *c, double *cx, double *cy, /* bool *changed, */ int k, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int cluster;
        double minDist = (INT_MAX); 
        // for really large values of k, the for loop creates a huge bottleneck
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

// single block, k many threads
__global__ void check_cluster_change(double *cx, double *cy, double *prev_cx, double *prev_cy, bool* changed, bool *red_change, int k) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadId > k){ return; }

    if (threadId == 0){ *red_change = false; }

    changed[threadId] = (cx[threadId] != prev_cx[threadId]) || (cy[threadId] != prev_cy[threadId]);

    __syncthreads();

    for (int s = 1; s < k; s *= 2) { 
        int index = 2 * s * threadId;
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
        *red_change |= changed[0];
    }

}


__global__ void updateKernel(int *x, int *y, int *c, int k, int n, double *sumx, double *sumy, /* bool* changed, bool* cont, */ int *count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&sumx[c[idx]], x[idx]);
        atomicAdd(&sumy[c[idx]], y[idx]);
        atomicAdd(&count[c[idx]], 1);
    }
}

__global__ void computeCentroids(int k, double *sumx, double *sumy, int *count, double *cx, double *cy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k) {
        if (count[idx] > 0) {
            cx[idx] = sumx[idx] / count[idx];
            cy[idx] = sumy[idx] / count[idx];
        } 
        #ifdef DEBUG
            else {
                printf("centroid %d: count=0\n", idx);
            }
        #endif
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

    cudaSetDevice(0); // titan X

    int iter = 0;
    int * count = new int[k];
    double acc_red_time = 0.0f, acc_assign_time = 0.0f, acc_update_time = 0.0f, acc_centroid_time = 0.0f;

    int *d_x, *d_y, *d_c;
    double *d_cx, *d_cy, *d_sumx, *d_sumy; 
    int *d_count;
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
    
    cudaMemcpy(d_x, x, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cx, cx, k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cy, cy, k * sizeof(double), cudaMemcpyHostToDevice);
    printf("Begin\n");

    #ifdef FLAT // WARNING: for large values of k assignKernel_flat returns wrong results due to contention over shared memory and other stuff... Don't use it!
        if (k <= 1024){
            printf("Flat\n");
        } else { 
            printf("Warning: cannot use FLAT with k > 1024, usign regular assign kernel instead\n");
        }
    #endif

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    cudaMemset(d_c, -1, n * sizeof(int)); 

    while (iter < MAX_ITER) {
        #ifdef DEBUG
            printf("==========%d=========\n", iter);
        #endif
        cont = false;
        
        cudaMemset(d_sumx,  0, k * sizeof(double));
        cudaMemset(d_sumy,  0, k * sizeof(double));
        cudaMemset(d_count, 0, k * sizeof(int));

        cudaEventRecord(start, 0);

        #ifdef FLAT // WARNING: for large values of k assignKernel_flat returns wrong results due to contention over shared memory and other stuff... Don't use it!
            if (k <= 1024){
                blockSize = k;
                // size_t sharedMemSize = blockSize * (sizeof(double) + sizeof(int));
                // assignKernel_flat_old<<<((n*k)+blockSize - 1)/blockSize, blockSize, sharedMemSize>>>(d_x, d_y, d_c, d_cx, d_cy, /*d_changed,*/ k, n);
                assignKernel_flat<<<((n*k)+blockSize - 1)/blockSize, blockSize>>>(d_x, d_y, d_c, d_cx, d_cy, /*d_changed,*/ k, n);
            } else {
                assignKernel<<<gridSize, blockSize>>>(d_x, d_y, d_c, d_cx, d_cy, /*d_changed,*/ k, n);
            }
        #else
            assignKernel<<<gridSize, blockSize>>>(d_x, d_y, d_c, d_cx, d_cy, /*d_changed,*/ k, n);
        #endif

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

        #ifdef HOSTREDUCE // WARNING: for large values of k HOSTREDUCE causes significant performance penalties... Don't use it! I left this here to obtain results to discuss in the report. 
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
        printf("HOST Acc Reduction: %f seconds\n", acc_red_time/1000);
        printf("Acc Data Transfer: %f seconds\n", acc_data_transfer_time/1000);
    #else
        printf("Acc Assign: %f seconds\n", acc_assign_time/1000);
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
    printf("Kmeans total runtime: %f seconds (Cuda events)\n", timer_overall/1000);
}

void kmeans_multigpu(int *x, int *y, int *c, double *cx, double *cy, int k, int n) {
    int gpu_ids[2] = {0, 1}; // weirdly titan x gpus are listed as indices 1 and 3 in nvidia-smi but are actually indices 0 and 1

    int gpu_count;
    cudaGetDeviceCount(&gpu_count);
    if (gpu_ids[0] >= gpu_count || gpu_ids[1] >= gpu_count) {
        printf("Requested GPUs not available\n");
        return;
    }

    cudaDeviceProp prop1, prop3;
    cudaGetDeviceProperties(&prop1, gpu_ids[0]);
    cudaGetDeviceProperties(&prop3, gpu_ids[1]);
    printf("Using GPUs: %d (%s) and %d (%s)\n", 
           gpu_ids[0], prop1.name, 
           gpu_ids[1], prop3.name);

    cudaEvent_t start[2], stop[2], overall_start, overall_stop;
    cudaStream_t streams[2];
    
    for (int gpu = 0; gpu < 2; gpu++) {
        cudaSetDevice(gpu_ids[gpu]);
        cudaEventCreate(&start[gpu]);
        cudaEventCreate(&stop[gpu]);
        cudaStreamCreate(&streams[gpu]);
    }
    cudaEventCreate(&overall_start);
    cudaEventCreate(&overall_stop);
    cudaEventRecord(overall_start, 0);

    int iter = 0;
    int * count = new int[k];
    // double acc_red_time = 0.0f, acc_assign_time = 0.0f, acc_update_time = 0.0f, acc_centroid_time = 0.0f;
    double acc_assign_time[2] = {0.0f, 0.0f};

    int n_per_gpu = n / 2;
    int remainder = n % 2;
    int n_gpu0 = n_per_gpu + remainder;
    int n_gpu1 = n_per_gpu;

    int *d_x[2], *d_y[2], *d_c[2];
    double *d_cx[2], *d_cy[2], *d_sumx[2], *d_sumy[2];
    int *d_count[2];
    bool cont = false;

    for (int gpu = 0; gpu < 2; gpu++) {
        cudaSetDevice(gpu_ids[gpu]);
        int current_n = (gpu == 0) ? n_gpu0 : n_gpu1;
        
        cudaMalloc(&d_x[gpu], current_n * sizeof(int));
        cudaMalloc(&d_y[gpu], current_n * sizeof(int));
        cudaMalloc(&d_c[gpu], current_n * sizeof(int));
        cudaMalloc(&d_cx[gpu], k * sizeof(double));
        cudaMalloc(&d_cy[gpu], k * sizeof(double));
        cudaMalloc(&d_sumx[gpu], k * sizeof(double));
        cudaMalloc(&d_sumy[gpu], k * sizeof(double));
        cudaMalloc(&d_count[gpu], k * sizeof(int));
    }

    cudaSetDevice(gpu_ids[0]);
    cudaMemcpy(d_x[0], x, n_gpu0 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y[0], y, n_gpu0 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cx[0], cx, k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cy[0], cy, k * sizeof(double), cudaMemcpyHostToDevice);

    cudaSetDevice(gpu_ids[1]);
    cudaMemcpy(d_x[1], x + n_gpu0, n_gpu1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y[1], y + n_gpu0, n_gpu1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cx[1], cx, k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cy[1], cy, k * sizeof(double), cudaMemcpyHostToDevice);

    printf("Begin\n");

    int blockSize = 256;
    int gridSize[2];
    gridSize[0] = (n_gpu0 + blockSize - 1) / blockSize;
    gridSize[1] = (n_gpu1 + blockSize - 1) / blockSize;

    for (int gpu = 0; gpu < 2; gpu++) {
        cudaSetDevice(gpu_ids[gpu]);
        cudaMemset(d_c[gpu], -1, (gpu == 0 ? n_gpu0 : n_gpu1) * sizeof(int));
    }

    while (iter < MAX_ITER) {
        cont = false;
        
        for (int gpu = 0; gpu < 2; gpu++) {
            cudaSetDevice(gpu_ids[gpu]);
            cudaMemset(d_sumx[gpu], 0, k * sizeof(double));
            cudaMemset(d_sumy[gpu], 0, k * sizeof(double));
            cudaMemset(d_count[gpu], 0, k * sizeof(int));
        }

        for (int gpu = 0; gpu < 2; gpu++) {
            cudaSetDevice(gpu_ids[gpu]);
            int current_n = (gpu == 0) ? n_gpu0 : n_gpu1;
            
            cudaEventRecord(start[gpu], streams[gpu]);
            assignKernel<<<gridSize[gpu], blockSize, 0, streams[gpu]>>>(
                d_x[gpu], d_y[gpu], d_c[gpu], d_cx[gpu], d_cy[gpu], k, current_n
            );
            cudaDeviceSynchronize();
            cudaEventRecord(stop[gpu], streams[gpu]);
            cudaEventSynchronize(stop[gpu]);
            float timer_assign = 0.0f;
            cudaEventElapsedTime(&timer_assign, start[gpu], stop[gpu]);
            acc_assign_time[gpu] += timer_assign;
            
            updateKernel<<<gridSize[gpu], blockSize, 0, streams[gpu]>>>(
                d_x[gpu], d_y[gpu], d_c[gpu], k, current_n, d_sumx[gpu], d_sumy[gpu], d_count[gpu]
            );
        }

        for (int gpu = 0; gpu < 2; gpu++) {
            cudaSetDevice(gpu_ids[gpu]);
            cudaDeviceSynchronize();
        }

        cudaSetDevice(gpu_ids[0]);
        double *h_sumx = new double[k];
        double *h_sumy = new double[k];
        int *h_count = new int[k];

        cudaMemcpy(h_sumx, d_sumx[0], k * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_sumy, d_sumy[0], k * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_count, d_count[0], k * sizeof(int), cudaMemcpyDeviceToHost);

        double *h_sumx1 = new double[k];
        double *h_sumy1 = new double[k];
        int *h_count1 = new int[k];

        cudaSetDevice(gpu_ids[1]);
        cudaMemcpy(h_sumx1, d_sumx[1], k * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_sumy1, d_sumy[1], k * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_count1, d_count[1], k * sizeof(int), cudaMemcpyDeviceToHost);

        for (int i = 0; i < k; i++) {
            h_sumx[i] += h_sumx1[i];
            h_sumy[i] += h_sumy1[i];
            h_count[i] += h_count1[i];
        }

        for (int gpu = 0; gpu < 2; gpu++) {
            cudaSetDevice(gpu_ids[gpu]);
            cudaMemcpy(d_sumx[gpu], h_sumx, k * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_sumy[gpu], h_sumy, k * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_count[gpu], h_count, k * sizeof(int), cudaMemcpyHostToDevice);

            // Compute new centroids
            computeCentroids<<<(k + blockSize - 1) / blockSize, blockSize, 0, streams[gpu]>>>(
                k, d_sumx[gpu], d_sumy[gpu], d_count[gpu], d_cx[gpu], d_cy[gpu]
            );
        }

        // Check convergence
        double *prev_cx = new double[k];
        double *prev_cy = new double[k];
        memcpy(prev_cx, cx, k * sizeof(double));
        memcpy(prev_cy, cy, k * sizeof(double));

        cudaSetDevice(gpu_ids[0]);
        cudaMemcpy(cx, d_cx[0], k * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(cy, d_cy[0], k * sizeof(double), cudaMemcpyDeviceToHost);

        for (int i = 0; i < k; i++) {
            cont |= ((cx[i] != prev_cx[i]) || (cy[i] != prev_cy[i]));
        }

        // Clean up temporary arrays
        delete[] h_sumx;
        delete[] h_sumy;
        delete[] h_count;
        delete[] h_sumx1;
        delete[] h_sumy1;
        delete[] h_count1;
        delete[] prev_cx;
        delete[] prev_cy;

        if (!cont) {
            // Copy final results back to host
            cudaSetDevice(gpu_ids[0]);
            cudaMemcpy(c, d_c[0], n_gpu0 * sizeof(int), cudaMemcpyDeviceToHost);
            cudaSetDevice(gpu_ids[1]);
            cudaMemcpy(c + n_gpu0, d_c[1], n_gpu1 * sizeof(int), cudaMemcpyDeviceToHost);
            
            printf("Converged at iteration %d\n", iter);
            writeClusterAssignments(x, y, c, n, OUTFILE);
            break;
        }

        iter++;
    }

    // Clean up
    for (int gpu = 0; gpu < 2; gpu++) {
        cudaSetDevice(gpu_ids[gpu]);
        cudaFree(d_x[gpu]);
        cudaFree(d_y[gpu]);
        cudaFree(d_c[gpu]);
        cudaFree(d_cx[gpu]);
        cudaFree(d_cy[gpu]);
        cudaFree(d_sumx[gpu]);
        cudaFree(d_sumy[gpu]);
        cudaFree(d_count[gpu]);
        cudaEventDestroy(start[gpu]);
        cudaEventDestroy(stop[gpu]);
        cudaStreamDestroy(streams[gpu]);
    }

    cudaEventRecord(overall_stop, 0);
    cudaEventSynchronize(overall_stop);
    float timer_overall;
    cudaEventElapsedTime(&timer_overall, overall_start, overall_stop);
    printf("Accumulated assign kernel runtime gpu 1: %f seconds\n", acc_assign_time[0]/1000);
    printf("Accumulated assign kernel runtime gpu 2: %f seconds\n", acc_assign_time[1]/1000);
    printf("Kmeans total runtime (Cuda events): %f seconds\n", timer_overall/1000);
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
    double init_begin = omp_get_wtime();
    randomCenters(x, y, n, k, cx, cy);
    double init_end = omp_get_wtime();
    printf("Random Centers Init Time (OMP wtime): %f seconds \n", init_end - init_begin);

    // Measure k-means execution time
    double kmeans_start = omp_get_wtime();
    #ifndef MULTIGPU
        kmeans(x, y, c, cx, cy, k, n);
    #else
        printf("Multi GPU\n");
        kmeans_multigpu(x, y, c, cx, cy, k, n);
    #endif

    double kmeans_end = omp_get_wtime();
    printf("K-Means Execution Time (OMP wtime): %f seconds \n", kmeans_end - kmeans_start);

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
