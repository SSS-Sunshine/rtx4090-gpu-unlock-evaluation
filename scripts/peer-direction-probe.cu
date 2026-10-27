#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#define CK(call) do { \
  cudaError_t error = (call); \
  if (error != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
    std::exit(1); \
  } \
} while (0)

__global__ void peer_write_kernel(uint4 *remote, const uint4 *local, size_t count) {
  size_t index = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * (size_t)blockDim.x;
  for (; index < count; index += stride) remote[index] = local[index];
}

__global__ void peer_read_kernel(uint4 *local, const uint4 *remote, size_t count) {
  size_t index = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * (size_t)blockDim.x;
  for (; index < count; index += stride) local[index] = remote[index];
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s LOCAL_GPU REMOTE_GPU read|write\n", argv[0]);
    return 2;
  }
  int local_gpu = std::atoi(argv[1]);
  int remote_gpu = std::atoi(argv[2]);
  bool write_mode = std::string(argv[3]) == "write";
  if (!write_mode && std::string(argv[3]) != "read") return 2;

  constexpr size_t bytes = 256ull << 20;
  constexpr int iterations = 40;
  size_t count = bytes / sizeof(uint4);
  int can = 0;
  CK(cudaDeviceCanAccessPeer(&can, local_gpu, remote_gpu));
  if (!can) return 3;

  CK(cudaSetDevice(local_gpu));
  cudaError_t peer = cudaDeviceEnablePeerAccess(remote_gpu, 0);
  if (peer != cudaSuccess && peer != cudaErrorPeerAccessAlreadyEnabled) CK(peer);
  cudaGetLastError();

  uint4 *local = nullptr, *remote = nullptr;
  CK(cudaSetDevice(local_gpu));
  CK(cudaMalloc(&local, bytes));
  CK(cudaMemset(local, 0x5a, bytes));
  CK(cudaSetDevice(remote_gpu));
  CK(cudaMalloc(&remote, bytes));
  CK(cudaMemset(remote, 0xa5, bytes));

  CK(cudaSetDevice(local_gpu));
  for (int i = 0; i < 5; ++i) {
    if (write_mode) peer_write_kernel<<<4096, 256>>>(remote, local, count);
    else peer_read_kernel<<<4096, 256>>>(local, remote, count);
  }
  CK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CK(cudaEventCreate(&start));
  CK(cudaEventCreate(&stop));
  CK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    if (write_mode) peer_write_kernel<<<4096, 256>>>(remote, local, count);
    else peer_read_kernel<<<4096, 256>>>(local, remote, count);
  }
  CK(cudaEventRecord(stop));
  CK(cudaEventSynchronize(stop));
  float milliseconds = 0;
  CK(cudaEventElapsedTime(&milliseconds, start, stop));
  double gbps = (double)bytes * iterations / (milliseconds / 1000.0) / 1e9;
  std::printf("local=%d remote=%d mode=%s payload_GBps=%.3f elapsed_ms=%.3f\n",
              local_gpu, remote_gpu, write_mode ? "write" : "read", gbps, milliseconds);

  CK(cudaSetDevice(local_gpu));
  CK(cudaFree(local));
  CK(cudaSetDevice(remote_gpu));
  CK(cudaFree(remote));
  return 0;
}
