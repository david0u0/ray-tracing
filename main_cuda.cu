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

    curandState st;
    curand_init(1984, 9, 0, &st);

    *world = HittableList(22 * 22 + 100);
    auto mat_ground = new Lambertian(Vec3{0.5, 0.5, 0.5});
    world->add(new Sphere(Vec3{0, -1000, 0}, 1000, mat_ground));
    for (int i = -11; i < 11; i++) {
        for (int j = -11; j < 11; j++) {
            Vec3 center(i + 0.7*rand_double({}, &st), 0.2, j + 0.7*rand_double({}, &st));
            if ((center - Vec3(4, 0.2, 0)).length() < 0.9) {
                continue;
            }

            double rand_mat = rand_double({}, &st);
            Material* mat;
            Vec3 albedo = {rand_double({}, &st), rand_double({}, &st), rand_double({}, &st)};
            if (rand_mat < 0.6) {
                // diffuse
                mat = new Lambertian(std::move(albedo));
            } else if (rand_mat < 0.85) {
                // metal
                auto fuse = rand_double({{0, 0.5}}, &st);
                mat = new Metal(std::move(albedo), fuse);
            } else {
                // glass
                mat = new Dielectric(Vec3{0.9, 0.9, 0.9}, 1.5);
            }
            world->add(new Sphere(center, 0.2, mat));
        }
    }

    auto material1 = new Dielectric(Vec3{1, 1, 1}, 1.5);
    world->add(new Sphere(Vec3{0, 1, 0}, 1.0, material1));

    auto material2 = new Lambertian(Vec3{0.4, 0.2, 0.1});
    world->add(new Sphere(Vec3{-4, 1, 0}, 1.0, material2));

    auto material3 = new Metal(Vec3{0.7, 0.6, 0.5}, 0.0);
    world->add(new Sphere(Vec3{4, 1, 0}, 1.0, material3));

    CamConfig conf;
    conf.width = 1200;
    conf.ratio = 16.0 / 9.0;

    conf.virtical_fov = 3.1415926535 / 9;
    conf.look_at = {0, 0, 0};
    conf.camera_center = {13, 2, 3};
    conf.samp_size = 30;

    conf.focus_dist = 10.0;
    conf.defocus_angle = 3.1415926 / 300;

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
    dim3 blocks(width/tx + 1, height/ty + 1);
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
