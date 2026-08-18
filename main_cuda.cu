#define CUDA

#include "common.cuh"
#include "world.cuh"

#include <iostream>

using namespace std;

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        cerr << "CUDA error = " << cudaGetErrorString(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ Vec3 calc_ray_color(const Ray &r) {
    Vec3 dir = r.dir;
    Vec3 unit = dir / dir.length();
    auto a = 0.5 * (unit.y() + 1.0);
    return (1.0 - a) * Vec3{1.0, 1.0, 1.0} + Vec3{0.5, 0.7, 1.0} * a;
}

__global__ void render(Vec3 *fb, int max_x, int max_y, Vec3 pixel000_loc, Vec3 pixel_delta_u, Vec3 pixel_delta_v, Vec3 camera_center) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= max_x || j >= max_y) {
        return;
    }

    auto pixel_center = pixel000_loc + (pixel_delta_u * i) + (pixel_delta_v * j);
    auto ray_dir = pixel_center - camera_center;
    Ray r(camera_center, ray_dir);
    int idx = j * max_x + i;
    fb[idx] = calc_ray_color(r);
}

int main() {
    int width = 800;
    int height = 500;
    int num_pixels = width * height;
    size_t fb_size = num_pixels * sizeof(Vec3);

    // allocate FB
    Vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // Camera
    double focal_length = 1.0;
    double viewport_height = 2.0;
    double viewport_width = viewport_height * (double(width) / height);
    auto camera_center = Vec3(0, 0, 0);

    //
    auto viewport_u = Vec3(viewport_width, 0, 0);
    auto viewport_v = Vec3(0, -viewport_height, 0);

    //
    auto pixel_delta_u = viewport_u / width;
    auto pixel_delta_v = viewport_v / height;

    //
    auto viewport_upper_left = camera_center - Vec3(0, 0, focal_length) - viewport_u/2 - viewport_v/2;
    auto pixel000_loc = viewport_upper_left + (pixel_delta_u + pixel_delta_v) * 0.5;

    // Render
    int tx = 8, ty = 8;
    dim3 blocks(width/ty + 1, height/ty + 1);
    dim3 threads(tx, ty);
    render<<<blocks, threads>>>(fb, width, height, pixel000_loc, pixel_delta_u, pixel_delta_v, camera_center);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    cout << "P3" << endl;
    cout << width << " " << height << endl;
    cout << 255 << endl;
    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            int idx = j * width + i;
            write_color(std::move(fb[idx]));
        }
    }
}
