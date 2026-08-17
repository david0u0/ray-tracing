#include "common.cuh"

#include <iostream>

using namespace std;

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void render(float *fb, int max_x, int max_y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= max_x || j >= max_y) {
        return;
    }
    int idx = 3 * (j * max_x + i);
    fb[idx + 0] = float(i) / max_x;
    fb[idx + 1] = float(j) / max_y;
    fb[idx + 2] = 0.2;
}

int main() {
    int width = 800;
    int height = 500;
    int num_pixels = width * height;
    size_t fb_size = 3 * num_pixels * sizeof(float);

    // allocate FB
    float *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    int tx = 8, ty = 8;
    dim3 blocks(width/ty + 1, height/ty + 1);
    dim3 threads(tx, ty);
    render<<<blocks, threads>>>(fb, width, height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    cout << "P3" << endl;
    cout << width << " " << height << endl;
    cout << 255 << endl;
    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            int idx = 3 * (j * width + i);
            write_color({ fb[idx], fb[idx + 1], fb[idx + 2] });
        }
    }
}
