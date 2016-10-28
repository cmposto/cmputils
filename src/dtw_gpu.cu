#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../include/cmputils/dtw.h"
#include "cuda_utils.h"

#include <iostream>
#include <string>
#include <cmath>
#include <chrono>

using us = std::chrono::microseconds;
using get_time = std::chrono::steady_clock;
using std::chrono::duration_cast;

#define dist_euclidean(x,y) ((x-y))

const int BLOCK_THREAD_MAX = 1024;

__global__
void initialize(const double *input, const int input_size, const double elem_0, double *T, int *W, const int stride, const int warp_window,
                const int scan_step, const int iteration){

    const int elem = threadIdx.x + iteration*BLOCK_THREAD_MAX;
    const int idx = threadIdx.x;

    extern __shared__ double S[];

    //init scan array
    if(elem<warp_window && elem<input_size) {
        S[idx] = std::abs(dist_euclidean(input[elem], elem_0));
        // for multi-block runs copy the previous value
        if((elem%BLOCK_THREAD_MAX==0) && (elem>0)){
            S[idx] = S[idx]+T[elem*stride-stride];
        }
    }
    __syncthreads();

    //inclusive scan
    bool phase = false;
    for (int step = 0; step < scan_step; step++) {
        int delta = 1 << step;
        int pair = idx - delta;
        S[(!phase)*blockDim.x+idx] = S[phase*blockDim.x+idx];
        if (pair >= 0) {
            S[(!phase)*blockDim.x+idx] = S[phase*blockDim.x+idx] + S[phase*blockDim.x+pair];
        }
        phase = !phase;
        __syncthreads();
    }

    //update temporary matrix
    if ((elem<warp_window) && (elem<input_size)) {
        int elem_idx = 0 + elem * stride;
        W[elem_idx] = elem;
        T[elem_idx] = S[idx+phase*blockDim.x];
    }
    __syncthreads();
}

__global__
void compute(const double *a, const double *b, double *T, double *R, int *WR, int *WC, const int size_a,
             const int size_b, const int warp_window, const int iteration){

    // copy row array into shared memory
    extern __shared__ double A[];

    const int copy_iterations = (size_a+blockDim.x-1)/blockDim.x;
    for(int i=0; i<copy_iterations; i++){
        int elem_idx = i*blockDim.x+threadIdx.x;
        if(elem_idx<size_a) {
            A[elem_idx] = a[elem_idx];
        }
    }
    __syncthreads();

    // copy column array into local memory
    const int col = threadIdx.x + iteration*BLOCK_THREAD_MAX;
    double col_value = 0;
    if(col>0 && col<size_b){
        col_value = b[col];
    }

    // start compute
    for(int elem = 1; elem<(size_a+size_b-1); elem++) {
        int row = elem - col;
        if(col>0 && col<size_b && row>0 && row<size_a) {
            double xy = std::abs(dist_euclidean(col_value, A[row]));
            int elem_idx = row * size_b + col;
            int elem_idx_x = (row - 1) * size_b + col - 1;
            int elem_idx_c = (row - 1) * size_b + col;
            int elem_idx_r = (row - 0) * size_b + col - 1;

            //TODO revise the global mem access

            //global reads
            double cost_x = T[elem_idx_x];
            double cost_c = T[elem_idx_c];
            double cost_r = T[elem_idx_r];
            double wr_r   = WR[elem_idx_r];
            double wc_c   = WC[elem_idx_c];

            double wr = 0;
            double wc = 0;
            double cost = 0;
            if ((cost_x <= cost_r) && (cost_x <= cost_c)) {
                cost = cost_x+xy;
            } else if ((cost_r < cost_c) && (wr_r < warp_window)) {
                cost = cost_r+xy;
                wr = wr_r+1;
            } else if (wc_c < warp_window) {
                cost = cost_c+xy;
                wc = wc_c+1;
            } else {
                cost = cost_x+xy;
            }

            //global writes
            WR[elem_idx] = wr;
            WC[elem_idx] = wc;
            T[elem_idx] = cost;
        }
        __syncthreads();
    }

    if(col==(size_b-1)){
        R[0] = T[size_a*size_b-1];
    }
}

double dtw(const double *a_in, const double *b_in, const int size_a_in, const int size_b_in, const int warp_window, const int verbose){
    // use b (column) the largest input array
    const double *a, *b;
    int size_a, size_b;
    if(size_a_in>size_b_in){
        b = a_in;
        a = b_in;
        size_a = size_b_in;
        size_b = size_a_in;
    } else {
        a = a_in;
        b = b_in;
        size_a = size_a_in;
        size_b = size_b_in;
    }

    //validate
    if(size_a*warp_window<size_b){
        int min_warp = size_b/size_a+1;
        throw std::runtime_error("Invalid input. Minimum warp window is: "+std::to_string(min_warp));
    }

    const size_t nElem = size_a*size_b;
    const size_t nBytes = nElem * sizeof(double);

    auto t0 = get_time::now();

    cudaFree(0);

    auto t1 = get_time::now();

    //result
    double h_R[1];
    double* h_T;

    if(verbose>1) {
        h_T = new double[nElem];
    }

    // allocate device
    double *d_A, *d_B, *d_T, *d_R;
    int *d_WC, *d_WR;
    checkCudaErrors(cudaMalloc(&d_A, size_a * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_B, size_b * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_T, nBytes));
    checkCudaErrors(cudaMalloc(&d_WR, nElem * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_WC, nElem * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_R, sizeof(double)));

    // initialize device values
    checkCudaErrors(cudaMemset(d_T, 127, nBytes));
    checkCudaErrors(cudaMemset(d_WR, 127, nElem * sizeof(int)));
    checkCudaErrors(cudaMemset(d_WC, 127, nElem * sizeof(int)));

    // copy memory from host to device
    checkCudaErrors(cudaMemcpy(d_A, a, size_a * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, b, size_b * sizeof(double), cudaMemcpyHostToDevice));

    auto t2 = get_time::now();

    int iterations = 0;
    int compute_total = 0;
    int block_thread_count = 0;
    int scan_step = 0;

    compute_total = warp_window;
    iterations = (compute_total + BLOCK_THREAD_MAX - 1) / BLOCK_THREAD_MAX;
    for(int iteration=0; iteration<iterations; iteration++) {
        block_thread_count = std::min<int>(BLOCK_THREAD_MAX, compute_total);
        compute_total -= block_thread_count;
        scan_step = std::ceil(log2(block_thread_count * 1.0));
        if(verbose>1){
            std::cout << "Iteration: " << iteration << " of: " << iterations << " warp window: " << warp_window << " scan step: " << scan_step  << " threads: " << block_thread_count << std::endl;
        }
        // initialize first row
        initialize <<< 1, block_thread_count, 2*block_thread_count*sizeof(double) >>> (d_B, size_b, a[0], d_T, d_WR, 1, warp_window, scan_step, iteration);
        // initialize first column
        initialize <<< 1, block_thread_count, 2*block_thread_count*sizeof(double) >>> (d_A, size_a, b[0], d_T, d_WC, size_b, warp_window, scan_step, iteration);
    }

    compute_total = size_b;
    iterations = (compute_total + BLOCK_THREAD_MAX - 1) / BLOCK_THREAD_MAX;
    for(int iteration=0; iteration<iterations; iteration++) {
        block_thread_count = std::min<int>(BLOCK_THREAD_MAX, compute_total);
        compute_total -= block_thread_count;
        if(verbose>1){
            std::cout << "Iteration: " << iteration << " of: " << iterations << " threads: " << block_thread_count << std::endl;
        }
        compute <<< 1, block_thread_count, size_a*sizeof(double) >>> (d_A, d_B, d_T, d_R, d_WR, d_WC, size_a, size_b, warp_window, iteration);
    }
    checkCudaErrors(cudaDeviceSynchronize());

    auto t3 = get_time::now();

    // copy result from device to host
    checkCudaErrors(cudaMemcpy(h_R, d_R, sizeof(double), cudaMemcpyDeviceToHost));

    if(verbose>1) {
        checkCudaErrors(cudaMemcpy(h_T, d_T, nBytes, cudaMemcpyDeviceToHost));

        std::cout << "Scan: " << std::endl;
        for(int r=0; r<size_a; r++){
            if(r<20 || r>(size_a-20)) {
                for (int c = 0; c < size_b; c++) {
                    int elem = r * size_b + c;
                    std::cout << h_T[elem] << ", ";
                }
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;

        //free host memory
        delete[] h_T;
    }

    // free device memory
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_T));
    checkCudaErrors(cudaFree(d_WR));
    checkCudaErrors(cudaFree(d_WC));
    checkCudaErrors(cudaFree(d_R));

    auto t4 = get_time::now();

    if(verbose>0) {
        std::cout << "INIT=" << duration_cast<us>(t1 - t0).count();
        std::cout << " ALLOCATE=" << duration_cast<us>(t2 - t1).count();
        std::cout << " COMPUTE=" << duration_cast<us>(t3 - t2).count();
        std::cout << " RESULT=" << duration_cast<us>(t4 - t3).count();
        std::cout << " TOTAL=" << duration_cast<us>(t4 - t1).count();
        std::cout << " DIST=" << h_R[0];
        std::cout << std::endl;
    }

    return h_R[0];
}
