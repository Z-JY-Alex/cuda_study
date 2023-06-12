#include "error.cuh"
#include <math.h>
#include <stdio.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
__global__ void add(const double *x, const double *y, double *z, const int N);
void check(const double *z, const int N);

int main()
{
    const int N = 100000001;
    const int M = sizeof(double) * N;
    double *h_x = (double*) malloc(M);
    double *h_y = (double*) malloc(M);
    double *h_z = (double*) malloc(M);

    for (int n=0; n<N; ++n){
        h_x[n] = a;
        h_y[n] = b;
    }

    double *d_x, *d_y, *d_z;
    CHECK(cudaMalloc((void**)&d_x, M));
    CHECK(cudaMalloc((void**)&d_y, M));
    CHECK(cudaMalloc((void**)&d_z, M));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));
    // _KERNER_CHECK用于檢測和函數。
    #ifdef KERNEL_CHECK
        const int block_size = 1280;
    #else
        const int block_size = 128;
    #endif

    const int grid_size = (N-1) / block_size + 1;
    add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    /// check kernel
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost));
    check(h_z, N);

    free(h_x);
    free(h_y);
    free(h_z);
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_z));
    return 0;
}

__global__ void add(const double *x, const double *y, double *z, const int N)
{
    const int cur_thread = blockDim.x * blockIdx.x + threadIdx.x;
    // 用於檢測内存
    #ifdef MEM_CHECK
        z[cur_thread] = x[cur_thread] + y[cur_thread];
    #else
        if (cur_thread < N)
        {
            z[cur_thread] = x[cur_thread] + y[cur_thread];
        }
    #endif
}

void check(const double *x, const int N)
{
    bool has_error = false;
    for (int n=0; n<N; ++n)
    {
        if (fabs(x[n]-c) > EPSILON){
            has_error = true;
        }
    }
    printf("%s\n", has_error?"Has Errors":"No errors");
}