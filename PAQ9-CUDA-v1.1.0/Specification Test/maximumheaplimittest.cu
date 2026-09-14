#include <iostream>
#include <cuda_runtime.h>

int main() {
    int device = 0;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        std::cerr << "Failed to get CUDA device properties.\n";
        return 1;
    }

    std::cout << "Device: " << prop.name << '\n';
    std::cout << "Total VRAM: " << prop.totalGlobalMem / (1024.0 * 1024 * 1024) << " GB\n\n";

    size_t low = 0;
    size_t high = prop.totalGlobalMem;
    size_t bestLimit = 0;

    // Binary search to find the maximum allowed heap size
    while (low <= high) {
        size_t mid = low + (high - low) / 2;
        
        cudaError_t err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, mid);
        
        if (err == cudaSuccess) {
            bestLimit = mid;
            low = mid + 1; // Try a larger size
        } else {
            // If high is at the max possible value to avoid underflow
            if (high == 0 || mid == 0) break;
            high = mid - 1; // Try a smaller size
        }
    }

    // Set the found maximum limit permanently for the session
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, bestLimit);

    std::cout << "Maximum Possible Heap Limit Found: " 
              << bestLimit / (1024.0 * 1024 * 1024) << " GB (" 
              << bestLimit << " bytes)\n";

    return 0;
}