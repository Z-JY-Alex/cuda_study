#include <stdio.h>

__global__ void hello_from_gpu()
{
    const int bid = blockIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    printf("hello world form block %d thread (%d, %d, %d)\n", bid, tx, ty, tz);
}

int main(){
    const dim3 block_size(2,2,2);
    hello_from_gpu<<<2, block_size>>>();
    cudaDeviceSynchronize();
    return 0;
}