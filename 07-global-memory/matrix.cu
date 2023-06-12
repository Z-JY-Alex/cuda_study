#include "../04-error-check/error.cuh"
#include <stdio.h>

#ifdef USE_DP
    using real = double;
#else
    using real = float;
#endif

const int NUM_REPEATS = 10;
const int TILE_DIM = 32;

void timing(const real *d_A, real *d_B, const int N, const int task);
__global__ void copy(const real *A, real *B, const int N);
__global__ void transpose1(const real *A, real *B, const int N);
__global__ void transpose2(const real *A, real *B, const int N);
__global__ void transpose3(const real *A, real *B, const int N);
void print_matrix(const int N, const real *A);

int main(int argc, char **argv)
{
    if (argc != 2){
        printf("usageL %s N\n", argv[0]);
        exit(1);
    }
    const int N = atoi(argv[1]);
    const int N2 = N*N;
    const int M = sizeof(real) * N2;
    real *h_A = (real *) malloc(M);
    real *h_B = (real *) malloc(M);
    for(int n=0; n<N2; ++n){
        h_A[n] = n;
    }
    real *d_A, *d_B;
    CHECK(cudaMalloc(&d_A, M));
    CHECK(cudaMalloc(&d_B, M));
    CHECK(cudaMemcpy(d_A, h_A, M, cudaMemcpyHostToDevice));

    printf("\ncopy:\n");
    timing(d_A, d_B, N, 0);
    printf("\ntranspose with coalesced read:\n");
    timing(d_A, d_B, N, 1);
    printf("\ntranspose with coalesced write:\n");
    timing(d_A, d_B, N, 2);
    printf("\ntranspose with coalesced write and __ldg read:\n");
    timing(d_A, d_B, N, 3);

    CHECK(cudaMemcpy(h_B, d_B, M, cudaMemcpyDeviceToHost));
    if (N <= 10)
    {
        printf("A =\n");
        print_matrix(N, h_A);
        printf("\nB =\n");
        print_matrix(N, h_B);
    }

    free(h_A);
    free(h_B);
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    return 0;
}


void timing(const real *d_A, real *d_B, const int N, const int task)
{
    const int grid_size_x = (N-1) / TILE_DIM +1;
    const int grid_size_y = grid_size_x;
    const dim3 block_size(TILE_DIM, TILE_DIM);
    const dim3 grid_size(grid_size_x, grid_size_y);
    float t_sum = 0;
    float t2_sum = 0;
    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        switch (task)
        {
            case 0:
                copy<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            case 1:
                transpose1<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            case 2:
                transpose2<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            case 3:
                transpose3<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            default:
                printf("Error: wrong task\n");
                exit(1);
                break;
        }

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        if (repeat > 0)
        {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("Time = %g +- %g ms.\n", t_ave, t_err);
}


__global__ void copy(const real *A, real *B, const int N)
{
    const int n_x = threadIdx.x + blockIdx.x*blockDim.x;
    const int n_y = threadIdx.y + blockIdx.y*blockDim.y;
    const int index = n_y*N + n_x;
    if (n_x < N && n_y < N)
    {
        B[index] = A[index];
    }
}

__global__ void transpose1(const real *A, real *B, const int N)
{
    const int n_x = threadIdx.x + blockIdx.x*blockDim.x;
    const int n_y = threadIdx.y + blockIdx.y*blockDim.y;
    if (n_x < N && n_y < N)
    {
        B[n_x*N+n_y] = A[n_y*N+n_x];  // 合并读，非合并写
    }
}

__global__ void transpose2(const real *A, real *B, const int N)
{
    const int n_x = threadIdx.x + blockIdx.x*blockDim.x;
    const int n_y = threadIdx.y + blockIdx.y*blockDim.y;
    if (n_x < N && n_y < N)
    {
        B[n_y*N+n_x] = A[n_x*N+n_y];   //  合并写，非合并读
    }
}

__global__ void transpose3(const real *A, real *B, const int N)
{
    const int n_x = threadIdx.x + blockIdx.x*blockDim.x;
    const int n_y = threadIdx.y + blockIdx.y*blockDim.y;
    if (n_x < N && n_y < N)
    {
        B[n_y*N+n_x] = __ldg(&A[n_x*N+n_y]);
    }
}


void print_matrix(const int N, const real *A)
{
    for (int n_y=0; n_y<N; ++n_y)
    {
        for (int n_x=0; n_x<N; ++n_x)
        {
            printf("%g\t", A[n_y*N+n_x]);
        }
        printf("\n");
    }
}