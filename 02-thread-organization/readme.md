1. cuda程序流程

   ```cpp
   int main()
   {
       主机函数
       核函数调用
       主机代码
       return 0;
   }
   ```
2. kerner 函数 前方需要__global__或者__device__前缀

   * __global__ 前缀函数称为全局(kernel)函数由主机端调用，在设备端多个线程块并行执行,且返回值只能是void
   * __device__ 前缀函数成为设备函数只能由kernel函数或者其他设备函数调用，返回值任意。
3. thread、block、grid

   * grid_size 指的是grid中所使用的block的尺寸,block_size指的是block中所使用的thread的尺寸。
   * grid_size在x, y, z方向上的大小限制分别为: $2^{31}-1$、65535、 65535
   * block_size在x, y, z方向上的限制为1024，1024，64且乘积*不能大于1024*
   * 计算当前线程ID

     ```cpp
     int nx = threadIdx.x + blockDim.x * blockIdx.x;
     int ny = threadIdx.y + blockDim.y * blockIdx.y;
     int nz = threadIdx.z + blockDim.z * blockIdx.z;
     ```
   * grid 和 block 可以用dim3定义维度，默认为1

     ```cpp
     dim3 grid_size(2, 2, 3)
     dim3 block_size(3, 3)
     ```
4. 核函数的调用

   * kernel_name<<`<grid_size, block_size>`>>(**param)
5. cudaDeviceSynchronize()

   * 阻塞主机进行，等待设备函数执行完毕后，继续执行。
6. 编译

   * 使用nvcc指令编译, 其中-arch指的是虚拟架构计算能力，code指的是真实计算能力，真实计算能力可以大于虚拟计算能力，通常设置一样即可。可以不写，不通cuda版本由不通的默认值，针对特定的卡的话，可以在nvcc官网查看显卡计算能力。

     ```shell
     nvcc -arch=compute_XY -code=sm_XY
     ```
