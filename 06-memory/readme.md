1. 局部内存、寄存器、共享内存、全局内存、常量内存、纹理内存、CPU内存直接的关系
2. 静态全局内存变量

   ```cpp
   __device__ T x;   单个变量
   __device__ T y[N];    固定长度变量
   ```

   * 静态全局内存变量，在核函数中可以直接调用。
   * 在主机函数中调用需要API

   ```cpp
   cudaMemcpyToSymbol()    //H2D
   cudaMemcpyFromSymbol()    //D2H
   ```
