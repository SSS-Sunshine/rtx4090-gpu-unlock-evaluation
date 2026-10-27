// Keep >32 GiB resident on one GPU while a small CUDA VMM allocation is
// mapped into both GPUs. The source GPU writes every word; the destination
// verifies every word and checks that the large local allocation is unchanged.
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define RT(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(e)); std::exit(1); } } while (0)

#define CU(call) do { CUresult e = (call); if (e != CUDA_SUCCESS) { \
  const char *name = nullptr, *text = nullptr; cuGetErrorName(e, &name); \
  cuGetErrorString(e, &text); std::fprintf(stderr, "CU error %s:%d: %s (%s)\n", \
  __FILE__, __LINE__, name ? name : "unknown", text ? text : "unknown"); \
  std::exit(1); } } while (0)

__global__ void fill_pattern(uint64_t *p, size_t n, uint64_t seed) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) p[i] = (i + seed) * 0x9E3779B97F4A7C15ull;
}

__global__ void check_pattern(const uint64_t *p, size_t n, uint64_t seed,
                              unsigned long long *bad) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n && p[i] != (i + seed) * 0x9E3779B97F4A7C15ull)
    atomicAdd(bad, 1ull);
}

__global__ void set_last(uint64_t *p, size_t n, uint64_t value) {
  if (blockIdx.x == 0 && threadIdx.x == 0) p[n - 1] = value;
}

static size_t round_up(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

int main(int argc, char **argv) {
  int src = argc > 1 ? std::atoi(argv[1]) : 0;
  int dst = argc > 2 ? std::atoi(argv[2]) : 1;
  size_t resident_gib = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 34;
  size_t window_mib = argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 64;
  if (src == dst || resident_gib <= 32 || window_mib == 0) return 2;

  RT(cudaSetDevice(src));
  RT(cudaFree(nullptr));
  CUcontext src_ctx = nullptr;
  CU(cuCtxGetCurrent(&src_ctx));
  RT(cudaSetDevice(dst));
  RT(cudaFree(nullptr));
  CUcontext dst_ctx = nullptr;
  CU(cuCtxGetCurrent(&dst_ctx));

  int can = 0;
  RT(cudaDeviceCanAccessPeer(&can, src, dst));
  if (!can) {
    std::fprintf(stderr, "no P2P from GPU%d to GPU%d\n", src, dst);
    return 2;
  }

  const size_t resident_bytes = resident_gib << 30;
  const size_t resident_words = resident_bytes / sizeof(uint64_t);
  const uint64_t resident_sentinel = 0xC0DEC0DE12345678ull;
  uint64_t *resident = nullptr;
  size_t free_before = 0, total = 0;
  RT(cudaSetDevice(dst));
  RT(cudaMemGetInfo(&free_before, &total));
  RT(cudaMalloc(&resident, resident_bytes));
  set_last<<<1, 1>>>(resident, resident_words, resident_sentinel);
  RT(cudaDeviceSynchronize());

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = dst;
  size_t granularity = 0;
  CU(cuMemGetAllocationGranularity(&granularity, &prop,
                                   CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  const size_t window_bytes = round_up(window_mib << 20, granularity);

  CUmemGenericAllocationHandle handle;
  CU(cuCtxSetCurrent(dst_ctx));
  CU(cuMemCreate(&handle, window_bytes, &prop, 0));

  CUdeviceptr dst_va = 0;
  CU(cuMemAddressReserve(&dst_va, window_bytes, granularity, 0, 0));
  CU(cuMemMap(dst_va, window_bytes, 0, handle, 0));
  CUmemAccessDesc dst_access = {};
  dst_access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  dst_access.location.id = dst;
  dst_access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CU(cuMemSetAccess(dst_va, window_bytes, &dst_access, 1));

  CUdeviceptr src_va = 0;
  CU(cuCtxSetCurrent(src_ctx));
  CU(cuMemAddressReserve(&src_va, window_bytes, granularity, 0, 0));
  CU(cuMemMap(src_va, window_bytes, 0, handle, 0));
  CUmemAccessDesc src_access = {};
  src_access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  src_access.location.id = src;
  src_access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CU(cuMemSetAccess(src_va, window_bytes, &src_access, 1));

  const size_t words = window_bytes / sizeof(uint64_t);
  const uint64_t seed = 0xA5A50000ull + (uint64_t)src * 257 + dst;
  RT(cudaSetDevice(src));
  fill_pattern<<<(words + 255) / 256, 256>>>(
      reinterpret_cast<uint64_t *>(src_va), words, seed);
  RT(cudaDeviceSynchronize());

  unsigned long long *device_bad = nullptr;
  RT(cudaSetDevice(dst));
  RT(cudaMalloc(&device_bad, sizeof(*device_bad)));
  RT(cudaMemset(device_bad, 0, sizeof(*device_bad)));
  check_pattern<<<(words + 255) / 256, 256>>>(
      reinterpret_cast<uint64_t *>(dst_va), words, seed, device_bad);
  RT(cudaDeviceSynchronize());
  unsigned long long bad = 0;
  RT(cudaMemcpy(&bad, device_bad, sizeof(bad), cudaMemcpyDeviceToHost));
  uint64_t sentinel_after = 0;
  RT(cudaMemcpy(&sentinel_after, resident + resident_words - 1,
                sizeof(sentinel_after), cudaMemcpyDeviceToHost));

  RT(cudaFree(device_bad));
  CU(cuCtxSetCurrent(src_ctx));
  CU(cuMemUnmap(src_va, window_bytes));
  CU(cuMemAddressFree(src_va, window_bytes));
  CU(cuCtxSetCurrent(dst_ctx));
  CU(cuMemUnmap(dst_va, window_bytes));
  CU(cuMemAddressFree(dst_va, window_bytes));
  CU(cuMemRelease(handle));
  RT(cudaSetDevice(dst));
  RT(cudaFree(resident));

  std::printf("src=GPU%d dst=GPU%d resident=%zuGiB window=%zuMiB "
              "free_before=%.2fGiB mismatches=%llu resident_sentinel=%s -> %s\n",
              src, dst, resident_gib, window_bytes >> 20,
              free_before / 1073741824.0, bad,
              sentinel_after == resident_sentinel ? "PASS" : "FAIL",
              bad == 0 && sentinel_after == resident_sentinel ? "PASS" : "FAIL");
  return bad == 0 && sentinel_after == resident_sentinel ? 0 : 1;
}
