#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CU(call) do { cudaError_t e=(call); if(e!=cudaSuccess){std::fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(2);} } while(0)

__device__ __forceinline__ uint64_t pattern(size_t i, uint64_t seed) {
    uint64_t x = (uint64_t)i ^ seed;
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
__global__ void write_pattern(uint64_t* data, size_t n, uint64_t seed) {
    for (size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=(size_t)gridDim.x*blockDim.x)
        data[i]=pattern(i,seed);
}
__global__ void verify_pattern(const uint64_t* data, size_t n, uint64_t seed, unsigned long long* errors) {
    unsigned long long local=0;
    for (size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=(size_t)gridDim.x*blockDim.x)
        local += data[i] != pattern(i,seed);
    if(local) atomicAdd(errors,local);
}
__global__ void copy_vec(uint4* dst, const uint4* src, size_t n) {
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x) dst[i]=src[i];
}

static float time_events(void (*launch)(void*), void* ctx, int reps) {
    cudaEvent_t a,b; CU(cudaEventCreate(&a)); CU(cudaEventCreate(&b));
    CU(cudaEventRecord(a)); for(int i=0;i<reps;i++) launch(ctx); CU(cudaEventRecord(b)); CU(cudaEventSynchronize(b));
    float ms=0; CU(cudaEventElapsedTime(&ms,a,b)); CU(cudaEventDestroy(a)); CU(cudaEventDestroy(b)); return ms/reps;
}
struct CopyCtx { void* dst; const void* src; size_t bytes; int kind; };
static void memcpy_launch(void* p) { auto*c=(CopyCtx*)p; CU(cudaMemcpyAsync(c->dst,c->src,c->bytes,(cudaMemcpyKind)c->kind)); }
struct KernelCopyCtx { uint4* dst; uint4* src; size_t n; };
static void kernel_copy_launch(void* p) { auto*c=(KernelCopyCtx*)p; copy_vec<<<4096,256>>>(c->dst,c->src,c->n); }

int main(){
    int device=0; CU(cudaSetDevice(device)); cudaDeviceProp prop{}; CU(cudaGetDeviceProperties(&prop,device));
    size_t free0,total; CU(cudaMemGetInfo(&free0,&total));
    const size_t GiB=1ULL<<30, MiB=1ULL<<20;
    size_t copy_bytes=4*GiB; void *a=nullptr,*b=nullptr; CU(cudaMalloc(&a,copy_bytes)); CU(cudaMalloc(&b,copy_bytes));
    CU(cudaMemset(a,0x5a,copy_bytes)); CU(cudaMemset(b,0,copy_bytes));
    CopyCtx d2d{b,a,copy_bytes,(int)cudaMemcpyDeviceToDevice}; for(int i=0;i<3;i++) memcpy_launch(&d2d); CU(cudaDeviceSynchronize());
    float d2d_ms=time_events(memcpy_launch,&d2d,12);
    KernelCopyCtx kc{(uint4*)b,(uint4*)a,copy_bytes/sizeof(uint4)}; for(int i=0;i<3;i++) kernel_copy_launch(&kc); CU(cudaDeviceSynchronize());
    float kernel_ms=time_events(kernel_copy_launch,&kc,12);
    CU(cudaFree(a)); CU(cudaFree(b));

    size_t pcie_bytes=512*MiB; void* host=nullptr; void* dev=nullptr; CU(cudaHostAlloc(&host,pcie_bytes,cudaHostAllocDefault)); CU(cudaMalloc(&dev,pcie_bytes));
    std::fill_n((unsigned char*)host,pcie_bytes,0x3c);
    CopyCtx h2d{dev,host,pcie_bytes,(int)cudaMemcpyHostToDevice}, d2h{host,dev,pcie_bytes,(int)cudaMemcpyDeviceToHost};
    for(int i=0;i<3;i++) memcpy_launch(&h2d); CU(cudaDeviceSynchronize()); float h2d_ms=time_events(memcpy_launch,&h2d,16);
    for(int i=0;i<3;i++) memcpy_launch(&d2h); CU(cudaDeviceSynchronize()); float d2h_ms=time_events(memcpy_launch,&d2h,16);
    CU(cudaFree(dev)); CU(cudaFreeHost(host));

    size_t free1; CU(cudaMemGetInfo(&free1,&total));
    size_t reserve=2*GiB; size_t test_bytes=(free1>reserve)?free1-reserve:free1*8/10; test_bytes=(test_bytes/(256*MiB))*(256*MiB);
    uint64_t* memory=nullptr; unsigned long long* errors=nullptr; CU(cudaMalloc(&errors,sizeof(*errors)));
    while(test_bytes>=8*GiB && cudaMalloc(&memory,test_bytes)!=cudaSuccess){ cudaGetLastError(); test_bytes-=256*MiB; }
    if(!memory){std::fprintf(stderr,"Unable to allocate VRAM test region\n");return 3;}
    std::vector<uint64_t> seeds={0ULL,0xffffffffffffffffULL,0x9e3779b97f4a7c15ULL};
    std::printf("{\n\"gpu\":\"%s\",\"compute_capability\":\"%d.%d\",\"total_MiB\":%.1f,\"initial_free_MiB\":%.1f,",
        prop.name,prop.major,prop.minor,total/(double)MiB,free0/(double)MiB);
    std::printf("\n\"d2d_memcpy_payload_GBps\":%.3f,\"d2d_memcpy_read_write_GBps\":%.3f,",
        copy_bytes/(d2d_ms/1000.0)/1e9,2.0*copy_bytes/(d2d_ms/1000.0)/1e9);
    std::printf("\n\"copy_kernel_payload_GBps\":%.3f,\"copy_kernel_read_write_GBps\":%.3f,",
        copy_bytes/(kernel_ms/1000.0)/1e9,2.0*copy_bytes/(kernel_ms/1000.0)/1e9);
    std::printf("\n\"pcie_h2d_GBps\":%.3f,\"pcie_d2h_GBps\":%.3f,\"vram_test_MiB\":%.1f,\"patterns\":[",
        pcie_bytes/(h2d_ms/1000.0)/1e9,pcie_bytes/(d2h_ms/1000.0)/1e9,test_bytes/(double)MiB);
    size_t words=test_bytes/sizeof(uint64_t); unsigned long long total_errors=0;
    for(size_t j=0;j<seeds.size();j++){
        CU(cudaMemset(errors,0,sizeof(*errors))); cudaEvent_t x,y; CU(cudaEventCreate(&x)); CU(cudaEventCreate(&y));
        CU(cudaEventRecord(x)); write_pattern<<<4096,256>>>(memory,words,seeds[j]); CU(cudaEventRecord(y)); CU(cudaEventSynchronize(y)); float write_ms;CU(cudaEventElapsedTime(&write_ms,x,y));
        CU(cudaEventRecord(x)); verify_pattern<<<4096,256>>>(memory,words,seeds[j],errors); CU(cudaEventRecord(y)); CU(cudaEventSynchronize(y)); float verify_ms;CU(cudaEventElapsedTime(&verify_ms,x,y));
        unsigned long long e=0;CU(cudaMemcpy(&e,errors,sizeof(e),cudaMemcpyDeviceToHost));total_errors+=e;
        std::printf("%s{\"seed\":\"0x%016llx\",\"errors\":%llu,\"write_GBps\":%.3f,\"verify_GBps\":%.3f}",j?",":"",(unsigned long long)seeds[j],e,test_bytes/(write_ms/1000.0)/1e9,test_bytes/(verify_ms/1000.0)/1e9);
        CU(cudaEventDestroy(x));CU(cudaEventDestroy(y));
    }
    std::printf("],\"total_memory_errors\":%llu}\n",total_errors);
    CU(cudaFree(memory));CU(cudaFree(errors));return total_errors?4:0;
}
