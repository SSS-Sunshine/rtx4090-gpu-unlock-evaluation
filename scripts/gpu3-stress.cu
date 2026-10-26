#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#define CU(x) do{auto e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));std::exit(2);}}while(0)
#define BL(x) do{auto e=(x);if(e!=CUBLAS_STATUS_SUCCESS){std::fprintf(stderr,"cuBLAS %d\n",(int)e);std::exit(2);}}while(0)
int main(int argc,char**argv){
 int seconds=argc>1?std::atoi(argv[1]):60,n=8192;size_t cells=(size_t)n*n;
 __half *a,*b;float*c;CU(cudaMalloc(&a,cells*2));CU(cudaMalloc(&b,cells*2));CU(cudaMalloc(&c,cells*4));CU(cudaMemset(a,1,cells*2));CU(cudaMemset(b,2,cells*2));
 cublasHandle_t h;BL(cublasCreate(&h));const float alpha=1,beta=0;long long iters=0;auto start=std::chrono::steady_clock::now();
 while(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<seconds){
  BL(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&alpha,a,CUDA_R_16F,n,b,CUDA_R_16F,n,&beta,c,CUDA_R_32F,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));iters++;
 }
 CU(cudaDeviceSynchronize());double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 std::printf("{\"seconds\":%.6f,\"iterations\":%lld,\"average_TFLOPS\":%.3f}\n",elapsed,iters,2.0*n*n*n*iters/elapsed/1e12);
 cublasDestroy(h);cudaFree(a);cudaFree(b);cudaFree(c);return 0;
}
