
#include <stdio.h>
#include <stdlib.h>
#include "assert.h" 
#include <time.h>
#include <sys/time.h>
// CUDA runtime
#include <cuda_runtime.h>
#include <cuda.h>



#define THREAD_PER_BLOCK 256

// transfer vector
#define FETCH_FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__global__ void add(float* a, float* b, float* c)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    c[idx] = a[idx] + b[idx];
}
__global__ void vec2_add(float* a, float* b, float* c)
{
    int idx = (threadIdx.x + blockIdx.x * blockDim.x)*2;
    //c[idx] = a[idx] + b[idx];
    float2 reg_a = FETCH_FLOAT2(a[idx]);
    float2 reg_b = FETCH_FLOAT2(b[idx]);
    float2 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    FETCH_FLOAT2(c[idx]) = reg_c;
}

__global__ void vec4_add(float* a, float* b, float* c)
{
    int idx = (threadIdx.x + blockIdx.x * blockDim.x)*4;
    //c[idx] = a[idx] + b[idx];
    float4 reg_a = FETCH_FLOAT4(a[idx]);
    float4 reg_b = FETCH_FLOAT4(b[idx]);
    float4 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;
    FETCH_FLOAT4(c[idx]) = reg_c;
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
    double flopsPerAdd = 1.0* N;

    float *a=(float *)malloc(N*sizeof(float));
    float *b=(float *)malloc(N*sizeof(float));
    float *out=(float *)malloc(N*sizeof(float));
    float *d_a;
    float *d_b;
    float *d_out;
    cudaMalloc((void **)&d_a,N*sizeof(float));
    cudaMalloc((void **)&d_b,N*sizeof(float));
    cudaMalloc((void **)&d_out,N*sizeof(float));
    float *res=(float *)malloc(N*sizeof(float));

    for(int i=0;i<N;i++){
        a[i]=1;
        b[i]=i;
        res[i]=a[i]+b[i];
    }

    cudaMemcpy(d_a,a,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,N*sizeof(float),cudaMemcpyHostToDevice);
    float msecTotal = 0;
    int nIter = 2000;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //base version 
    dim3 Grid( N/THREAD_PER_BLOCK, 1);
    dim3 Block( THREAD_PER_BLOCK, 1);
    cudaEventRecord(start);
    for(int i=0; i<nIter; i++){
        add<<<Grid,Block>>>(d_a, d_b, d_out);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);

    cudaMemcpy(out,d_out,N*sizeof(float),cudaMemcpyDeviceToHost);
    
    double msecPerAdd = 0;
    double gigaFlops = 0;
    msecPerAdd = msecTotal / nIter;
    gigaFlops = (flopsPerAdd * 1.0e-9f) / (msecPerAdd / 1000.0f);
    printf( "Base version Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops,
        msecPerAdd,
        flopsPerAdd);
    if(check(out,res,N))printf("Base version add: Pass\n");
    else{
        printf("Base version add:Fail\n");
    }

    //vec2 
    dim3 Grid_vec2( N/THREAD_PER_BLOCK/2, 1);
    dim3 Block_vec2( THREAD_PER_BLOCK, 1);
    cudaEventRecord(start);
    for(int i=0; i<nIter; i++){
        vec2_add<<<Grid_vec2,Block_vec2>>>(d_a, d_b, d_out);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);

    cudaMemcpy(out,d_out,N*sizeof(float),cudaMemcpyDeviceToHost);
    
    double msecPerVec2 = 0;
    double gigaFlops_vec2 = 0;
    msecPerVec2 = msecTotal / nIter;
    gigaFlops_vec2 = (flopsPerAdd * 1.0e-9f) / (msecPerVec2 / 1000.0f);
    printf( "Vec4 Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops_vec2,
        msecPerVec2,
        flopsPerAdd);
    if(check(out,res,N))printf("Vec2 add: Pass\n");
    else{
        printf("Vec2 add:Fail\n");
    }

    //vec4 
    dim3 Grid_vec4( N/THREAD_PER_BLOCK/4, 1);
    dim3 Block_vec4( THREAD_PER_BLOCK, 1);
    cudaEventRecord(start);
    for(int i=0; i<nIter; i++){
        vec4_add<<<Grid_vec4,Block_vec4>>>(d_a, d_b, d_out);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);

    cudaMemcpy(out,d_out,N*sizeof(float),cudaMemcpyDeviceToHost);
    double msecPerVec4 = 0;
    double gigaFlops_vec4 = 0;
    msecPerVec4 = msecTotal / nIter;
    gigaFlops_vec4 = (flopsPerAdd * 1.0e-9f) / (msecPerVec4 / 1000.0f);
    printf( "Vec4 Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops_vec4,
        msecPerVec4,
        flopsPerAdd);
    if(check(out,res,N))printf("Vec4 add: Pass\n");
    else{
        printf("Vec4 add:Fail\n");
    }
      

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    free(a);
    free(b);
    free(out);
    free(res);
}
