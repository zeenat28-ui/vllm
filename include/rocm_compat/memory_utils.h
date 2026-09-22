#pragma once

#include <hip/hip_runtime.h>
#include <cstdlib>

namespace rocm_compat {
namespace memory {

inline void init_environment() noexcept {
    setenv("TMPDIR", "/tmp", 0);
}

inline hipError_t safe_malloc(void** ptr, size_t size) noexcept {
    const size_t aligned_size = (size + 255) & ~255;
    const hipError_t status = ::hipMalloc(ptr, aligned_size);
    if (status != hipSuccess) {
        *ptr = nullptr;
    }
    return status;
}

inline void safe_free(void* ptr) noexcept {
    if (ptr) {
        ::hipFree(ptr);
    }
}

} // namespace memory
} // namespace rocm_compat

__attribute__((constructor)) inline void initialize_rocm_compat_module() {
    rocm_compat::memory::init_environment();
}

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
#define hipMalloc(ptr, size) rocm_compat::memory::safe_malloc(ptr, size)
#define hipFree(ptr) rocm_compat::memory::safe_free(ptr)
#endif
