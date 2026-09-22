#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace rocm_compat {
namespace atomics {

__device__ __forceinline__ __half atomicAdd(__half* address, __half val) {
    unsigned int* const base_address = reinterpret_cast<unsigned int*>(reinterpret_cast<size_t>(address) & ~2);
    unsigned int long_val = *base_address;
    unsigned int assumed;
    __half old_val;

    do {
        assumed = long_val;
        if (reinterpret_cast<size_t>(address) & 2) {
            old_val.data = static_cast<unsigned short>(long_val >> 16);
            __half sum = __hadd(old_val, val);
            unsigned int new_val = (long_val & 0x0000FFFF) | (static_cast<unsigned int>(sum.data) << 16);
            long_val = ::atomicCAS(base_address, assumed, new_val);
        } else {
            old_val.data = static_cast<unsigned short>(long_val & 0x0000FFFF);
            __half sum = __hadd(old_val, val);
            unsigned int new_val = (long_val & 0xFFFF0000) | static_cast<unsigned int>(sum.data);
            long_val = ::atomicCAS(base_address, assumed, new_val);
        }
    } while (assumed != long_val);

    return old_val;
}

} // namespace atomics
} // namespace rocm_compat

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
#ifndef __HIP_NO_HALF_ATOMICS__
#define atomicAdd(addr, val) rocm_compat::atomics::atomicAdd(addr, val)
#endif
#endif
