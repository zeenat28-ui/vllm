#pragma once
#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>

namespace ROCmCompat {
    namespace Memory {
        inline void setup_safe_environment() {
            setenv("TMPDIR", "/tmp", 0);
        }
        inline hipError_t safe_hipMalloc(void** ptr, size_t size) {
            size_t aligned_size = (size + 255) & ~255; 
            hipError_t status = ::hipMalloc(ptr, aligned_size);
            if (status != hipSuccess) {
                *ptr = nullptr;
            }
            return status;
        }
        inline void safe_hipFree(void* ptr) {
            if (ptr != nullptr) {
                ::hipFree(ptr);
            }
        }
    }
}
__attribute__((constructor)) inline void init_rocm_compat() {
    ROCmCompat::Memory::setup_safe_environment();
    std::cout << "[ROCmCompatLib] Initialized: Memory Variables Patched.\n";
}
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
    #define hipMalloc(ptr, size) ROCmCompat::Memory::safe_hipMalloc(ptr, size)
    #define hipFree(ptr) ROCmCompat::Memory::safe_hipFree(ptr)
#endif
