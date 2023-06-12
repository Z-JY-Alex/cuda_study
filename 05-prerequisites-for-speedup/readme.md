1. cuda事件计时

   ```cpp
   cudaEvent_t start, end;
   cudaEventCreate(&start);
   cudaEventCreate(&end);
   cudaEventRecord(start);
   cudaEventQuery(start);

   需要计时代码 

   cudaEventRecord(end);
   cudaEventSynchronize(end);
   float elapsed_time;
   cudaEventElapsedTime(&elapsed_time, start, end);
   printf("Time = %g ms \n", elapsed_time);
   cudaEventDestory(start);
   cudaEventDestory(end);
   ```
2. cuda 程序高性能条件

   * 减少数据传输
   * 提高核函数算术强度
   * 增大核函数并行规模
