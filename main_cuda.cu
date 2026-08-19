#include <chrono>
#define CUDA

#include "common.cuh"
#include "world.cuh"
#include "camera.cuh"

#include <iostream>
#include <curand_kernel.h>

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

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= max_x || j >= max_y) {
        return;
    }
    int idx = j*max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, idx, 0, &rand_state[idx]);
}
__global__ void render(Vec3 *fb, Camera *camera, HittableList* world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= camera->width || j >= camera->height) {
        return;
    }

    int idx = j * camera->width + i;
    fb[idx] = camera->render(*world, i, j, &rand_state[idx]);
}

__global__ void init_world(HittableList *world, Camera *camera) {
    if (threadIdx.x != 0 && blockIdx.x != 0) {
        return;
    }

    *world = HittableList(4);
    auto mat_center = new Lambertian(Vec3{0.2, 0.2, 0.2});
    auto mat_ground = new Lambertian(Vec3{0.5, 0.5, 0.5});
    auto mat_left = new Dielectric(Vec3{1, 1, 1}, 1.0/1.3);
    auto mat_right = new Metal(Vec3{0.9, 1, 0.7}, 0.8);

    world->add(new Sphere(Vec3{0, 0, -1.2}, 0.5, mat_center));
    world->add(new Sphere(Vec3{1, 0, -1}, 0.5, mat_left));
    world->add(new Sphere(Vec3{-1, 0, -1}, 0.5, mat_right));
    world->add(new Sphere(Vec3{0, -100.5, -1}, 100, mat_ground));

    CamConfig conf;
    conf.width = 1200;
    conf.ratio = 16.0 / 9.0;
    conf.camera_center = {2, 1, 2};
    *camera = Camera(conf);
}

int main() {
    int width = 1200;
    int height = 675;
    int num_pixels = width * height;
    size_t fb_size = num_pixels * sizeof(Vec3);

    Vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    Camera *camera;
    checkCudaErrors(cudaMallocManaged((void **)&camera, sizeof(Camera)));
    HittableList *world;
    checkCudaErrors(cudaMallocManaged((void **)&world, sizeof(HittableList)));
    init_world<<<1, 1>>>(world, camera);
    checkCudaErrors(cudaGetLastError());

    int tx = 8, ty = 8;
    dim3 blocks(width/ty + 1, height/ty + 1);
    dim3 threads(tx, ty);

    // Random
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
    render_init<<<blocks, threads>>>(width, height, d_rand_state);

    // Render
    render<<<blocks, threads>>>(fb, camera, world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    cout << "P3" << endl;
    cout << width << " " << height << endl;
    cout << 255 << endl;
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            int idx = j * width + i;
            write_color(std::move(fb[idx]));
        }
    }
}
