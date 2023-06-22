#include <stdio.h>
#include <stdlib.h>
#include "assert.h" 
#include <time.h>
#include <sys/time.h>
// CUDA runtime
#include <cuda_runtime.h>
#include <cuda.h>

#define THREAD_PER_BLOCK 256

__global__ void reduce0(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = d_in[i];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}
//no_divergence_branch
__global__ void reduce1(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = d_in[i];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}
//no bank conflict
__global__ void reduce2(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = d_in[i];
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

// dim3 Grid( N/(2*THREAD_PER_BLOCK),1);
// adding during loading data to shared memory
__global__ void reduce3(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = d_in[i] + d_in[i + blockDim.x];
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}


__device__ void warpReduce(volatile float* cache, unsigned int tid){
    cache[tid]+=cache[tid+32];
    //__syncthreads();
    cache[tid]+=cache[tid+16];
    //__syncthreads();
    cache[tid]+=cache[tid+8];
    //__syncthreads();
    cache[tid]+=cache[tid+4];
    //__syncthreads();
    cache[tid]+=cache[tid+2];
    //__syncthreads();
    cache[tid]+=cache[tid+1];
    //__syncthreads();
}

__global__ void reduce4(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = d_in[i] + d_in[i + blockDim.x];
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid < 32) warpReduce(sdata, tid);
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}


template <unsigned int blockSize>
__device__ void warpReduce(volatile float* cache, unsigned int tid){
    if (blockSize >= 64)cache[tid]+=cache[tid+32];
    if (blockSize >= 32)cache[tid]+=cache[tid+16];
    if (blockSize >= 16)cache[tid]+=cache[tid+8];
    if (blockSize >= 8)cache[tid]+=cache[tid+4];
    if (blockSize >= 4)cache[tid]+=cache[tid+2];
    if (blockSize >= 2)cache[tid]+=cache[tid+1];
}

template <unsigned int blockSize>
__global__ void reduce5(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = d_in[i] + d_in[i + blockDim.x];
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) {
        if (tid < 256) { 
            sdata[tid] += sdata[tid + 256]; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 256) {
        if (tid < 128) { 
            sdata[tid] += sdata[tid + 128]; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 128) {
        if (tid < 64) { 
            sdata[tid] += sdata[tid + 64]; 
        } 
        __syncthreads(); 
    }
    if (tid < 32) warpReduce<blockSize>(sdata, tid);

    // write result for this block to global mem
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize, int NUM_PER_THREAD>
__global__ void reduce6(float *d_in,float *d_out, const int n){
    __shared__ float sdata[blockSize];

    // each thread loads NUM_PER_THREAD element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * NUM_PER_THREAD) + threadIdx.x;

    sdata[tid] = 0;

    #pragma unroll
    for(int iter=0; iter<NUM_PER_THREAD; iter++){
        sdata[tid] += d_in[i+iter*blockSize];
    }
    
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) {
        if (tid < 256) { 
            sdata[tid] += sdata[tid + 256]; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 256) {
        if (tid < 128) { 
            sdata[tid] += sdata[tid + 128]; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 128) {
        if (tid < 64) { 
            sdata[tid] += sdata[tid + 64]; 
        } 
        __syncthreads(); 
    }
    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    
    // write result for this block to global mem
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

bool check(float *out,float *res,int n){
    for(int i=0;i<n;i++){
        if(out[i]!=res[i])
            return false;
    }
    return true;
}

int main(){
    const int N=32*1024*1024;
    float *a=(float *)malloc(N*sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a,N*sizeof(float));

    int block_num=N/THREAD_PER_BLOCK;
    float *out=(float *)malloc((N/THREAD_PER_BLOCK)*sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out,(N/THREAD_PER_BLOCK)*sizeof(float));
    float *res=(float *)malloc((N/THREAD_PER_BLOCK)*sizeof(float));

    for(int i=0;i<N;i++){
        a[i]=1;
    }

    for(int i=0;i<block_num;i++){
        float cur=0;
        for(int j=0;j<THREAD_PER_BLOCK;j++){
            cur+=a[i*THREAD_PER_BLOCK+j];
        }
        res[i]=cur;
    }

    cudaMemcpy(d_a,a,N*sizeof(float),cudaMemcpyHostToDevice);
    
    float msecTotal = 0;
    int nIter = 2000;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 Grid( N/THREAD_PER_BLOCK,1);
    dim3 Block( THREAD_PER_BLOCK,1);

    float msecPerReductionV0 = 0;
    cudaEventRecord(start);
    for(int i=0; i<nIter; i++){
       reduce0<<<Grid,Block>>>(d_a,d_out);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);

    cudaMemcpy(out,d_out,block_num*sizeof(float),cudaMemcpyDeviceToHost);

    if(check(out,res,block_num))printf("V0: the ans is right\n");
    else{
        printf("V0: the ans is wrong\n");
    }
    
    msecPerReductionV0 = msecTotal / nIter;
    printf( "Base version Time= %.3f msec\n",msecPerReductionV0);

    float msecPerReductionV1 = 0;
    cudaEventRecord(start);
    for(int i=0; i<nIter; i++){
       reduce1<<<Grid,Block>>>(d_a,d_out);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);

    cudaMemcpy(out,d_out,block_num*sizeof(float),cudaMemcpyDeviceToHost);

    if(check(out,res,block_num))printf("V1: the ans is right\n");
    else{
        printf("V1: the ans is wrong\n");
    }
    
    msecPerReductionV1 = msecTotal / nIter;
    printf( "No divergence branch Version Time= %.3f msec, speeding ratio = %.3f \n",
             msecPerReductionV1, 
             msecPerReductionV0/msecPerReductionV1);

    float msecPerReductionV2 = 0;
    cudaEventRecord(start);
    for(int i=0; i<nIter; i++){
       reduce2<<<Grid,Block>>>(d_a,d_out);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);

    cudaMemcpy(out,d_out,block_num*sizeof(float),cudaMemcpyDeviceToHost);

    if(check(out,res,block_num))printf("V2: the ans is right\n");
    else{
        printf("V2: the ans is wrong\n");
    }
    
    msecPerReductionV2 = msecTotal / nIter;
    printf( "No bank confilct Version Time= %.3f msec,speeding ratio = %.3f \n",
            msecPerReductionV2,
            msecPerReductionV0/msecPerReductionV2);

    int NUM_PER_BLOCK = 2*THREAD_PER_BLOCK;
    block_num = N / NUM_PER_BLOCK;

    free(res);
    res=(float *)malloc(block_num*sizeof(float));
    for(int i=0;i<block_num;i++){
        float cur=0;
        for(int j=0;j<NUM_PER_BLOCK;j++){
            cur+=a[i * NUM_PER_BLOCK + j];
        }
        res[i]=cur;
    }

    dim3 Grid_V3( block_num, 1);
    dim3 Block_V3( THREAD_PER_BLOCK, 1);

    float msecPerReductionV3 = 0;
    cudaEventRecord(start);
    for(int i=0; i<nIter; i++){
       reduce3<<<Grid_V3,Block_V3>>>(d_a,d_out);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);

    cudaMemcpy(out,d_out,block_num*sizeof(float),cudaMemcpyDeviceToHost);

    if(check(out,res,block_num))printf("V3: the ans is right\n");
    else{
        printf("V3: the ans is wrong\n");
    }
    
    msecPerReductionV3 = msecTotal / nIter;
    printf( "Adding during laoding data Version Time= %.3f msec,speeding ratio = %.3f \n",
            msecPerReductionV3,
            msecPerReductionV0/msecPerReductionV3);

    //v4, unroll last warp 
    float msecPerReductionV4 = 0;
    cudaEventRecord(start);
    for(int i=0; i<nIter; i++){
       reduce4<<<Grid_V3,Block_V3>>>(d_a,d_out);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);

    cudaMemcpy(out,d_out,block_num*sizeof(float),cudaMemcpyDeviceToHost);

    if(check(out,res,block_num))printf("V4: the ans is right\n");
    else{
        printf("V4: the ans is wrong\n");
    }
    
    msecPerReductionV4 = msecTotal / nIter;
    printf( "Unroll last warp Version Time= %.3f msec,speeding ratio = %.3f \n",
            msecPerReductionV4,
            msecPerReductionV0/msecPerReductionV4);

    //v5, unroll completely 
    float msecPerReductionV5 = 0;
    cudaEventRecord(start);
    for(int i=0; i<nIter; i++){
       reduce5<THREAD_PER_BLOCK><<<Grid_V3,Block_V3>>>(d_a,d_out);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);

    cudaMemcpy(out,d_out,block_num*sizeof(float),cudaMemcpyDeviceToHost);

    if(check(out,res,block_num))printf("V5: the ans is right\n");
    else{
        printf("V5: the ans is wrong\n");
    }
    
    msecPerReductionV5 = msecTotal / nIter;
    printf( "Unroll completely Version Time= %.3f msec,speeding ratio = %.3f \n",
            msecPerReductionV5,
            msecPerReductionV0/msecPerReductionV5);


    //v6, multi add 
    const int block_v6 = 1024;
    const int NUM_PER_BLOCK_v6 = N / block_v6;
    const int NUM_PER_THREAD = NUM_PER_BLOCK_v6/THREAD_PER_BLOCK;

    free(out);
    out=(float *)malloc(block_v6*sizeof(float));

    cudaFree(d_out);
    cudaMalloc((void **)&d_out,block_v6*sizeof(float));

    free(res);
    res=(float *)malloc(block_v6*sizeof(float));
    for(int i=0;i<block_v6;i++){
        float cur=0;
        for(int j=0;j<NUM_PER_BLOCK_v6;j++){
            if(i * NUM_PER_BLOCK_v6 + j < N){
                cur+=a[i * NUM_PER_BLOCK_v6 + j];
            }
        }
        res[i]=cur;
    }

    dim3 Grid_v6( block_v6, 1);
    dim3 Block_v6( THREAD_PER_BLOCK, 1);
    float msecPerReductionV6 = 0;
    cudaEventRecord(start);
    for(int i=0; i<nIter; i++){
       reduce6<THREAD_PER_BLOCK, NUM_PER_THREAD><<<Grid_v6,Block_v6>>>(d_a,d_out,N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);

    cudaMemcpy(out,d_out,block_v6*sizeof(float),cudaMemcpyDeviceToHost);

    if(check(out,res,block_v6))printf("V6: the ans is right\n");
    else{
        printf("V6: the ans is wrong\n");
    }
    
    msecPerReductionV6 = msecTotal / nIter;
    printf( "Multli add Version Time= %.3f msec,speeding ratio = %.3f \n",
            msecPerReductionV6,
            msecPerReductionV0/msecPerReductionV6);

    cudaFree(d_a);
    cudaFree(d_out);

    free(a);
    free(out);
    free(res);

    return 0;
}
