#include <iostream>
#include <cuda_runtime.h>

int main() {
    size_t free_byte = 0;
    size_t total_byte = 0;

    cudaError_t err = cudaMemGetInfo(&free_byte, &total_byte);
    
    if (err != cudaSuccess) {
        std::cerr << "Failed to get memory info: " << cudaGetErrorString(err) << '\n';
        return 1;
    }

    double free_gb = (double)free_byte / (1024.0 * 1024 * 1024);
    double total_gb = (double)total_byte / (1024.0 * 1024 * 1024);

    std::cout << "Free Physical VRAM: " << free_gb << " GB (" << free_byte << " bytes)\n";
    std::cout << "Total Physical VRAM: " << total_gb << " GB (" << total_byte << " bytes)\n";

    return 0;
}