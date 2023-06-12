#include "../04-error-check/error.cuh"
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

void timing(real *h_x, real *d_x, const int method);

const int NUM_REPEATS = 100;
const int N = 100000000;
const int M = sizeof(real) * N;

const int block_size = 128;

int main(void)
{
    real *h_x = (real *) malloc(M);
    for (int n=0; n<N; ++n)
    {
        h_x[n] = 1.23;
    }
    real *d_x;
    CHECK(cudaMalloc(&d_x, M));

    printf("\nUsing global memory only:\n");
    timing(h_x, d_x, 0);
    printf("\nUsing static shared memory:\n");
    timing(h_x, d_x, 1);
    printf("\nUsing dynamic shared memory:\n");
    timing(h_x, d_x, 2);
    free(h_x);
    CHECK(cudaFree(d_x));
    return 0;
}

__global__ void reduce_global(real *d_x, real *d_y)
{
    const int tid = threadIdx.x;
    real *x = &d_x[blockIdx.x*blockDim.x];

    for(int offset=blockDim.x >>1; offset>0; offset>>=1)
    {
        if (tid < offset)
        {
            x[tid] += x[tid+offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d_y[blockIdx.x] = x[0];
    }
}


__global__ void reduce_shared(real *d_x, real *d_y)
{
    __shared__ real s_y[128];
    const int tid = threadIdx.x;
    const int n = tid + blockIdx.x*blockDim.x;
    s_y[tid] = (n<N) ? d_x[n] : 0.0;
    __syncthreads();

    for(int offset=blockDim.x>>1; offset>0; offset>>=1)
    {
        if(tid < offset)
        {
            s_y[tid] = s_y[tid+offset];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        d_y[blockIdx.x] = s_y[0];
    }
}


__global__ void reduce_dynamic(real *d_x, real *d_y)
{
    extern __shared__ real s_y[];
    const int tid = threadIdx.x;
    const int n = tid + blockIdx.x*blockDim.x;
    s_y[tid] = (n<N) ? d_x[n] : 0.0;
    __syncthreads();

    for(int offset=blockDim.x>>1; offset>0; offset>>=1)
    {
        if (tid < offset)
        {
            s_y[tid] = s_y[tid+offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d_y[blockIdx.x] = s_y[0];
    }
}


real reduce(real *d_x, const int method)
{
    int grid_size = (N-1) / block_size +1;
    const int ymem = sizeof(real) * grid_size;
    real *d_y;
    CHECK(cudaMalloc(&d_y, ymem));
    real *h_y = (real *) malloc(ymem);

    switch(method)
    {
        case 0:
            reduce_global<<<grid_size, block_size>>>(d_x, d_y);
            break;
        case 1:
            reduce_shared<<<grid_size, block_size>>>(d_x, d_y);
            break;
        case 2:
            reduce_dynamic<<<grid_size, block_size, sizeof(real)*block_size>>>(d_x, d_y);
            break;
        default:
            printf("Error: wrong method\n");
            exit(1);
            break;
    }
    CHECK(cudaMemcpy(h_y, d_y, ymem, cudaMemcpyDeviceToHost));

    real result = 0.0;
    for (int n=0; n<grid_size; ++n)
    {
        result += h_y[n];
    }
    free(h_y);
    CHECK(cudaFree(d_y));
    return result;
}

void timing(real *h_x, real *d_x, const int method)
{
    real sum = 0;

    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        sum = reduce(d_x, method);

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    printf("sum = %f.\n", sum);
}