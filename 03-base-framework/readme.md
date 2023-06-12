1. CUDA程序基本框架

   ```cpp
   int main()
   {
       分配主机内存
       主机内存数据初始化
       分配设备内存
       主机内存数据复制至设备
       执行kernel函数计算
       计算结果从设备内存拷贝至主机内存
       释放设备与主机内存
   }
   ```
2. grid_size, block_size大小定义

   * ```
     grid_size = (N-1)/block_size +1
     ```
3. 常用的cuda API

   ```cpp
   cudaMalloc()
   cudaMemcpy()
   cudaFree()
   ```
