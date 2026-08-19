#include <chrono>
#define CUDA

#include "common.cuh"
#include "world.cuh"
// #include "camera.cuh"

#include <iostream>
#include <curand_kernel.h>

const int samp_size = 20;
const double samp_scale = 1.0 / samp_size;

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

__device__ Vec3 calc_ray_color(Ray cur_ray, const HittableList &world, curandState *st) {
    Vec3 cur_color = {1, 1, 1};

    for (int i = 0; i < 30; i++) {
        auto rec = world.hit(cur_ray, {0.001, 99999});
        if (rec.has_value()) {
            auto scatter_rec = rec->mat->scatter(cur_ray, *rec, st);
            if (!scatter_rec.has_value()) {
                return {0, 0, 0};
            }
            cur_color = scatter_rec->attenuation * cur_color;
            cur_ray = scatter_rec->ray;
            continue;
        }

        Vec3 dir = cur_ray.dir;
        Vec3 unit = dir / dir.length();
        auto a = 0.5 * (unit.y() + 1.0);
        auto bg_color = (1.0 - a) * Vec3{1.0, 1.0, 1.0} + Vec3{0.5, 0.7, 1.0} * a;
        return cur_color * bg_color;
    }

    return {0, 0, 0};
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
__global__ void render(Vec3 *fb, int max_x, int max_y, Vec3 pixel000_loc, Vec3 pixel_delta_u, Vec3 pixel_delta_v, Vec3 camera_center, HittableList* world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= max_x || j >= max_y) {
        return;
    }

    int idx = j * max_x + i;
    Vec3 color;
    curandState local_rand_state = rand_state[idx];
    for (int k = 0; k < samp_size; k++) {
        double off_x = rand_double({}, &local_rand_state) - 0.5;
        double off_y = rand_double({}, &local_rand_state) - 0.5;
        auto pixel_center = pixel000_loc + (pixel_delta_u * (i + off_x)) + (pixel_delta_v * (j + off_y));
        auto ray_dir = pixel_center - camera_center;
        Ray r(camera_center, ray_dir);
        color = color + calc_ray_color(r, *world, &local_rand_state);
    }
    fb[idx] = color * samp_scale;
}

__global__ void render2(Vec3 *fb, int max_x, int max_y, Vec3 pixel000_loc, Vec3 pixel_delta_u, Vec3 pixel_delta_v, Vec3 camera_center, HittableList* world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= max_x || j >= max_y) {
        return;
    }

    auto pixel_center = pixel000_loc + (pixel_delta_u * i) + (pixel_delta_v * j);
    auto ray_dir = pixel_center - camera_center;
    Ray r(camera_center, ray_dir);
    int idx = j * max_x + i;
    fb[idx] = calc_ray_color(r, *world, &rand_state[idx]);
}

__global__ void init_hit_list(HittableList *world) {
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
}

int main() {
    int width = 1200;
    int height = 675;
    int num_pixels = width * height;
    size_t fb_size = num_pixels * sizeof(Vec3);

    Vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    HittableList *world;
    checkCudaErrors(cudaMallocManaged((void **)&world, sizeof(HittableList)));
    init_hit_list<<<1, 1>>>(world);
    checkCudaErrors(cudaGetLastError());

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

    int tx = 8, ty = 8;
    dim3 blocks(width/ty + 1, height/ty + 1);
    dim3 threads(tx, ty);

    // Random
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
    render_init<<<blocks, threads>>>(width, height, d_rand_state);

    // Render
    render<<<blocks, threads>>>(fb, width, height, pixel000_loc, pixel_delta_u, pixel_delta_v, camera_center, world, d_rand_state);
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
