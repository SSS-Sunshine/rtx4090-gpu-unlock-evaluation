#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#define CK(x) do {auto e=(x); if(e!=cudaSuccess) {fprintf(stderr,"CUDA_ERROR %s %s\n",#x,cudaGetErrorString(e));return 100;}}while(0)
constexpr size_t G=16384; // 64 KiB guard at each end
constexpr uint32_t SENTINEL=0x65abc39du;
__host__ __device__ uint32_t pat(size_t i,uint32_t seed) {return (uint32_t)i*0x9e3779b9u ^ seed;}
__global__ void initialize(uint32_t *p,size_t n,uint32_t seed,int patterned) {
 for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n+2*G;i+=blockDim.x*gridDim.x)
   p[i]=(patterned && i>=G && i<G+n)?pat(i-G,seed):SENTINEL;
}
__global__ void peerwrite(uint32_t *p,size_t n,uint32_t seed) {
 for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x)p[G+i]=pat(i,seed);
 __threadfence_system();
}
__global__ void peerread(const uint32_t *p,uint32_t *out,size_t n) {
 for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x)out[G+i]=p[G+i];
}
__global__ void verify(const uint32_t *p,size_t n,uint32_t seed,unsigned *errors) {
 for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n+2*G;i+=blockDim.x*gridDim.x) {
   uint32_t expected=(i>=G && i<G+n)?pat(i-G,seed):SENTINEL;
   if(p[i]!=expected)atomicAdd(errors,1u);
 }
}
int check(uint32_t *p,size_t n,uint32_t seed,unsigned *errors,unsigned *hosterrors) {
 CK(cudaMemset(errors,0,sizeof(unsigned)));
 verify<<<256,256>>>(p,n,seed,errors);CK(cudaGetLastError());CK(cudaDeviceSynchronize());
 CK(cudaMemcpy(hosterrors,errors,sizeof(unsigned),cudaMemcpyDeviceToHost));return 0;
}
int pair(int a,int b,int rounds) {
 int can=0;CK(cudaDeviceCanAccessPeer(&can,a,b));if(!can){printf("UNSUPPORTED %d %d\n",a,b);return 3;}
 CK(cudaSetDevice(a));CK(cudaDeviceEnablePeerAccess(b,0));
 for(int round=0;round<rounds;round++) for(size_t bytes: {4096ul,65536ul,1048576ul,16777216ul,67108864ul}) {
   size_t n=bytes/4,alloc=(n+2*G)*4;
   uint32_t seed=0x100003u+a*131u+b*17u+round*0x10001u;
   uint32_t *target,*out;unsigned *eowner,*ewriter,werr=0,rerr=0,keep=0;
   CK(cudaSetDevice(b));CK(cudaMalloc(&target,alloc));CK(cudaMalloc(&eowner,4));
   initialize<<<256,256>>>(target,n,seed,0);CK(cudaGetLastError());CK(cudaDeviceSynchronize());
   CK(cudaSetDevice(a));CK(cudaMalloc(&out,alloc));CK(cudaMalloc(&ewriter,4));
   initialize<<<256,256>>>(out,n,seed,0);CK(cudaGetLastError());CK(cudaDeviceSynchronize());
   peerwrite<<<256,256>>>(target,n,seed);CK(cudaGetLastError());CK(cudaDeviceSynchronize());
   CK(cudaSetDevice(b));int rc=check(target,n,seed,eowner,&werr);if(rc)return rc;
   if(werr){printf("FAIL_WRITE %d %d %zu errors=%u\n",a,b,bytes,werr);return 10;}
   seed^=0xa5f019c7u;
   initialize<<<256,256>>>(target,n,seed,1);CK(cudaGetLastError());CK(cudaDeviceSynchronize());
   CK(cudaSetDevice(a));peerread<<<256,256>>>(target,out,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());
   rc=check(out,n,seed,ewriter,&rerr);if(rc)return rc;
   CK(cudaSetDevice(b));rc=check(target,n,seed,eowner,&keep);if(rc)return rc;
   printf("%s src=%d dst=%d round=%d bytes=%zu write_errors=%u read_errors=%u owner_after_read_errors=%u\n",
          (werr||rerr||keep)?"FAIL":"PASS",a,b,round,bytes,werr,rerr,keep);fflush(stdout);
   if(rerr||keep)return 11;
   CK(cudaFree(eowner));CK(cudaFree(target));CK(cudaSetDevice(a));CK(cudaFree(ewriter));CK(cudaFree(out));
 }
 CK(cudaSetDevice(a));CK(cudaDeviceDisablePeerAccess(b));return 0;
}
int main(int argc,char **argv) {
 int n=0;CK(cudaGetDeviceCount(&n));
 int rounds=argc>1?atoi(argv[1]):1;if(rounds<1||rounds>20)return 2;
 if(argc==4){int a=atoi(argv[2]),b=atoi(argv[3]);if(a<0||b<0||a>=n||b>=n||a==b)return 2;return pair(a,b,rounds);}
 for(int a=0;a<n;a++)for(int b=0;b<n;b++)if(a!=b){int rc=pair(a,b,rounds);if(rc)return rc;}
 printf("MATRIX_PASS directed_pairs=%d rounds=%d sizes=5\n",n*(n-1),rounds);return 0;
}
