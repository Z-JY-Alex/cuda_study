#include <stdio.h>
#include <math.h>

const double EPSION = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

__global__ void add(const double *x, const double *y, double *z);
void check(const double *z, const int N);

int main()
{
    const int N = 100000000;
    const int M = sizeof(double) * N;
    double *h_x = new double[N];
    double *h_y = new double[N];
    double *h_z = new double[N];

    for (int n=0; n<N; ++n){
        h_x[n] = a;
        h_y[n] = b;
    }

    double *d_x, *d_y, *d_z;
    cudaMalloc(&d_x, M);
    cudaMalloc(&d_y, M);
    cudaMalloc(&d_z, M);
    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);

    const int block_size = 128;
    const int grid_size = N / block_size;

    add<<<grid_size, block_size>>>(d_x, d_y, d_z);
    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);

    check(h_z, N);
    
    delete h_x;
    delete h_y;
    delete h_z;
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    return 0;
}

__global__ void add(const double *x, const double *y, double *z)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    z[n] = x[n] + y[n];
}

void check(const double *z, const int N)
{
    bool has_error = false;
    for (int n=0; n<N; ++n)
    {
        if (fabs(z[n] - c) > EPSION)
        {
            has_error = true;
        }
    }
    printf("%s \n", has_error ? "Has Error":"No Error");
}