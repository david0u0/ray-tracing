#include "camera.cuh"
#include "world.cuh"
#include "common.cuh"

#include <iostream>
#include <math.h>
#include <memory>
#include <cstdlib>

int main() {
    Camera camera(CamConfig{
        .width = 400,
        .ratio = 16.0 / 9.0,
        .virtical_fov = 3.1415926535 / 9,
        .camera_center = {-2, 2, 1},
        .samp_size = 100,

        .focus_dist = 3.4,
        .defocus_angle = 3.1415926 / 18,
    });

    cout << "P3" << endl;
    cout << camera.width << " " << camera.height << endl;
    cout << 255 << endl;

    // World
    auto mat_ground = make_shared<Lambertian>(Vec3{0.8, 0.8, 0});
    auto mat_center = make_shared<Lambertian>(Vec3{0.1, 0.2, 0.5});
    auto mat_left = make_shared<Dielectric>(Vec3(1.0, 1.0, 1.0), 1.5);
    auto mat_bubble = make_shared<Dielectric>(Vec3(1.0, 1.0, 1.0), 1.0 / 1.5);
    auto mat_right = make_shared<Metal>(Vec3(0.8, 0.6, 0.2), 1);
    HittableList hit_list;
    hit_list.add(make_shared<Sphere>(Vec3{0, -100.5, -1}, 100, mat_ground));
    hit_list.add(make_shared<Sphere>(Vec3{0, 0, -1.2}, 0.5, mat_center));
    hit_list.add(make_shared<Sphere>(Vec3{-1, 0, -1}, 0.5, mat_left));
    hit_list.add(make_shared<Sphere>(Vec3{-1, 0, -1}, 0.4, mat_bubble));
    hit_list.add(make_shared<Sphere>(Vec3{1, 0, -1}, 0.5, mat_right));

    camera.render(hit_list);
}
