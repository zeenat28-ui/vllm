#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace ROCmCompat {
    namespace Atomics {
        __device__ __forceinline__ __half atomicAdd_half(__half* address, __half val) {
            unsigned int* base_address = (unsigned int*)((size_t)address & ~2);
            unsigned int long_val = *base_address;
            unsigned int assumed;
            __half old_val;
            do {
                assumed = long_val;
                if ((size_t)address & 2) {
                    old_val = __ushort_as_half((unsigned short)(long_val >> 16));
                    __half sum = __hadd(old_val, val);
                    unsigned int new_val = (long_val & 0x0000FFFF) | (((unsigned int)__half_as_ushort(sum)) << 16);
                    long_val = ::atomicCAS(base_address, assumed, new_val);
                } else {
                    old_val = __ushort_as_half((unsigned short)(long_val & 0x0000FFFF));
                    __half sum = __hadd(old_val, val);
                    unsigned int new_val = (long_val & 0xFFFF0000) | ((unsigned int)__half_as_ushort(sum));
                    long_val = ::atomicCAS(base_address, assumed, new_val);
                }
            } while (assumed != long_val);
            return old_val;
        }
    }
}
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
    #define atomicAdd(addr, val) ROCmCompat::Atomics::atomicAdd_half(addr, val)
#endif
