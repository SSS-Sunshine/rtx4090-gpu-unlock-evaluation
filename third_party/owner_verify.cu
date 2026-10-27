// Guarded probe: the allocation OWNER runs the validation kernel.
// --local runs safe self-tests on every GPU; --peer A B tests one direction.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) { \
 fprintf(stderr,"CUDA_ERROR %s: %s\n",#x,cudaGetErrorString(e)); return 100; }} while(0)
constexpr size_t WORDS = 16*1024*1024/sizeof(uint32_t);
constexpr size_t CENTER = WORDS/2;
constexpr unsigned PAYLOAD = 256;
constexpr uint32_t GUARD=0x5a3cc3a5u;
__host__ __device__ uint32_t pattern(unsigned i) { return 0xc0ffee01u ^ (i*0x9e3779b9u); }
struct Result { unsigned matches, guard_errors, found_count, found_word; };
__global__ void init(uint32_t *p) {
 for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<WORDS;i+=gridDim.x*blockDim.x) p[i]=GUARD;
}
__global__ void write_payload(uint32_t *p) {
 unsigned i=threadIdx.x;
 if(i<PAYLOAD) p[CENTER+i]=pattern(i);
 __threadfence_system();
}
__global__ void read_payload(const uint32_t *p,uint32_t *out) {
 unsigned i=threadIdx.x;
 if(i<PAYLOAD) out[i]=p[CENTER+i];
}
__global__ void verify_owner(const uint32_t *p,Result *r) {
 for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<WORDS;i+=gridDim.x*blockDim.x) {
   if(i>=CENTER && i<CENTER+PAYLOAD) {
     if(p[i]==pattern(i-CENTER)) atomicAdd(&r->matches,1u);
   } else if(p[i]!=GUARD) atomicAdd(&r->guard_errors,1u);
   if(p[i]==pattern(0) && i+PAYLOAD<=WORDS) {
     bool found=true;
     for(unsigned j=1;j<PAYLOAD;j++) if(p[i+j]!=pattern(j)) {found=false; break;}
     if(found) {atomicAdd(&r->found_count,1u); atomicExch(&r->found_word,(unsigned)i);}
   }
 }
}
int run(int writer,int owner,bool peer) {
 if(peer) {
   int can=0; CK(cudaDeviceCanAccessPeer(&can,writer,owner));
   if(!can) {printf("UNSUPPORTED writer=%d owner=%d\n",writer,owner); return 3;}
   CK(cudaSetDevice(writer));
   auto e=cudaDeviceEnablePeerAccess(owner,0);
   if(e!=cudaSuccess && e!=cudaErrorPeerAccessAlreadyEnabled) {printf("ENABLE_FAIL %s\n",cudaGetErrorString(e));return 4;}
   cudaGetLastError();
 }
 uint32_t *target=nullptr,*readback=nullptr;
 Result *device_result=nullptr,result{};
 CK(cudaSetDevice(owner));
 CK(cudaMalloc(&target,WORDS*sizeof(uint32_t)));
 CK(cudaMalloc(&device_result,sizeof(Result)));
 CK(cudaMemset(device_result,0,sizeof(Result)));
 init<<<256,256>>>(target); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
 CK(cudaSetDevice(writer));
 CK(cudaMalloc(&readback,PAYLOAD*sizeof(uint32_t)));
 write_payload<<<1,PAYLOAD>>>(target); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
 read_payload<<<1,PAYLOAD>>>(target,readback); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
 uint32_t host_readback[PAYLOAD];
 CK(cudaMemcpy(host_readback,readback,sizeof(host_readback),cudaMemcpyDeviceToHost));
 unsigned writer_matches=0;
 for(unsigned i=0;i<PAYLOAD;i++) writer_matches+=host_readback[i]==pattern(i);
 CK(cudaFree(readback));
 CK(cudaSetDevice(owner));
 verify_owner<<<256,256>>>(target,device_result); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
 CK(cudaMemcpy(&result,device_result,sizeof(result),cudaMemcpyDeviceToHost));
 bool good=writer_matches==PAYLOAD && result.matches==PAYLOAD && result.guard_errors==0
     && result.found_count==1 && result.found_word==CENTER;
 printf("%s writer=%d owner=%d writer_matches=%u owner_kernel_matches=%u guard_errors=%u found_count=%u found_byte=0x%x expected_byte=0x%zx delta=%lld\n",
     good?"PASS":"FAIL",writer,owner,writer_matches,result.matches,result.guard_errors,
     result.found_count,result.found_word*4,CENTER*4,((long long)result.found_word-(long long)CENTER)*4);
 CK(cudaFree(device_result)); CK(cudaFree(target));
 if(peer) {CK(cudaSetDevice(writer)); CK(cudaDeviceDisablePeerAccess(owner));}
 return good?0:10;
}
int main(int argc,char **argv) {
 int count=0; CK(cudaGetDeviceCount(&count));
 if(argc==2 && !strcmp(argv[1],"--local")) {
   for(int i=0;i<count;i++) {int rc=run(i,i,false); if(rc)return rc;}
   return 0;
 }
 if(argc==4 && !strcmp(argv[1],"--peer")) {
   int a=atoi(argv[2]),b=atoi(argv[3]);
   if(a<0||b<0||a>=count||b>=count||a==b)return 2;
   return run(a,b,true);
 }
 fprintf(stderr,"usage: %s --local | --peer WRITER OWNER\n",argv[0]);return 2;
}
