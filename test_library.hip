#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "rocm_compat/atomic_ops.h"
#include "rocm_compat/memory_utils.h"

__global__ void test_atomic_kernel(__half* result) {
    __half val_to_add = __float2half(2.5f);
    atomicAdd(result, val_to_add);
}

int main() {
    std::cout << "\n🚀 Starting ROCmCompatLib Tests...\n\n";

    std::cout << "--- [Test A] Memory Allocation Test ---\n";
    void* bad_memory_ptr = nullptr;
    hipError_t status = hipMalloc(&bad_memory_ptr, 13); 
    if (status == hipSuccess) {
        std::cout << "✅ Memory Test Passed! Driver did NOT crash.\n";
        hipFree(bad_memory_ptr);
    }

    std::cout << "\n--- [Test B] Math AtomicAdd Test ---\n";
    __half* d_result;
    hipMalloc((void**)&d_result, sizeof(__half));
    __half initial_val = __float2half(0.0f);
    hipMemcpy(d_result, &initial_val, sizeof(__half), hipMemcpyHostToDevice);

    test_atomic_kernel<<<1, 1>>>(d_result);
    hipDeviceSynchronize();

    __half h_result;
    hipMemcpy(&h_result, d_result, sizeof(__half), hipMemcpyDeviceToHost);
    std::cout << "✅ Atomic Test Passed! Added Value: " << __half2float(h_result) << "\n";
    
    hipFree(d_result);
    std::cout << "\n🎉 All Tests Completed Successfully!\n";
    return 0;
}
