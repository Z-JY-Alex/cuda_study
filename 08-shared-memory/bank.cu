#include "../04-error-check/error.cuh"
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 10;
const int TILE_DIM = 32;

void timing(const real *d_A, real *d_B, const int N, const int task);
__global__ void transpose1(const real *A, real *B, const int N);
__global__ void transpose2(const real *A, real *B, const int N);
void print_matrix(const int N, const real *A);

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("usage: %s \n", argv[0]);
        exit(1);
    }
    const int N = atoi(argv[1]);
    const int N2 = N*N;
    const int M = sizeof(real) * N2;
    real *h_A = (real *) malloc(M);
    real *h_B = (real *) malloc(M);
    for (int n=0; n<N2; ++n)
    {
        h_A[n] = n;
    }
    real *d_A, *d_B;
    CHECK(cudaMalloc(&d_A, M));
    CHECK(cudaMalloc(&d_B, M));
    CHECK(cudaMemcpy(d_A, h_A, M, cudaMemcpyHostToDevice));

    printf("\ntranspose with shared memory bank conflict:\n");
    timing(d_A, d_B, N, 1);
    printf("\ntranspose without shared memory bank conflict:\n");
    timing(d_A, d_B, N, 2);

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
    dim3 block_size(TILE_DIM, TILE_DIM);
    dim3 grid_size(grid_size_x, grid_size_y);
    float t_sum = 0;
    float t2_sum = 0;
    for(int repeat=0; repeat<=NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, end;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&end));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        switch (task)
        {
        case 1:
            transpose1<<<grid_size, block_size>>>(d_A, d_B, N);
            break;
        case 2:
            transpose2<<<grid_size, block_size>>>(d_A, d_B, N);
            break;
        default:
            break;
        }
        CHECK(cudaEventRecord(end));
        CHECK(cudaEventSynchronize(end));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, end));
        printf("Time = %g ms.\n", elapsed_time);
        if(repeat > 0)
        {
            t_sum +=elapsed_time;
            t2_sum += elapsed_time;
        }
        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(end));
    }
    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("Time = %g +- %g ms.\n", t_ave, t_err);
}


__global__ void transpose1(const real *A, real *B, const int N)
{
    __shared__ real S[TILE_DIM][TILE_DIM];
    int bx = blockIdx.x*blockDim.x;
    int by = blockIdx.y*blockDim.y;
    int nx1 = bx + threadIdx.x;
    int ny1 = by + threadIdx.y;
    if (nx1 < N && ny1 <N)
    {
        S[threadIdx.y][threadIdx.x] = A[ny1*N+nx1];
    }
    __syncthreads();
    int nx2 = bx + threadIdx.y;
    int ny2 = by + threadIdx.x;
    if (nx2 < N && ny2 < N)
    { 
        B[nx2*N + ny2] = S[threadIdx.x][threadIdx.y];
    }
}

__global__ void transpose2(const real *A, real *B, const int N)
{
    __shared__ real S[TILE_DIM][TILE_DIM+1];
    int bx = blockIdx.x*blockDim.x;
    int by = blockIdx.y*blockDim.y;
    int nx1 = bx + threadIdx.x;
    int ny1 = by + threadIdx.y;
    if (nx1 < N && ny1 <N)
    {
        S[threadIdx.y][threadIdx.x] = A[ny1*N+nx1];
    }
    __syncthreads();
    int nx2 = bx + threadIdx.y;
    int ny2 = by + threadIdx.x;
    if (nx2 < N && ny2 < N)
    {
        B[nx2*N + ny2] = S[threadIdx.x][threadIdx.y];
    }
}


void print_matrix(const int N, const real *A)
{
    for (int ny = 0; ny < N; ny++)
    {
        for (int nx = 0; nx < N; nx++)
        {
            printf("%g\t", A[ny * N + nx]);
        }
        printf("\n");
    }
}