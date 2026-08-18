#include "camera.cuh"
#include "world.cuh"
#include "common.cuh"

#include <iostream>
#include <math.h>
#include <memory>
#include <cstdlib>

int main() {
    Camera camera(CamConfig{
        .width = 1200,
        .ratio = 16.0 / 9.0,
        .virtical_fov = 3.1415926535 / 9,
        .look_at = {0, 0, 0},
        .camera_center = {13, 2, 3},
        .samp_size = 30,

        .focus_dist = 10.0,
        .defocus_angle = 3.1415926 / 300,
    });

    cout << "P3" << endl;
    cout << camera.width << " " << camera.height << endl;
    cout << 255 << endl;

    // World
    HittableList hit_list;
    auto mat_ground = make_shared<Lambertian>(Vec3{0.5, 0.5, 0.5});
    hit_list.add(make_shared<Sphere>(Vec3{0, -1000, 0}, 1000, mat_ground));

    for (int i = -11; i < 11; i++) {
        for (int j = -11; j < 11; j++) {
            Vec3 center(i + 0.7*rand_double({}), 0.2, j + 0.7*rand_double({}));
            if ((center - Vec3(4, 0.2, 0)).length() < 0.9) {
                continue;
            }

            double rand_mat = rand_double({});
            shared_ptr<Material> mat;
            Vec3 albedo = {rand_double({}), rand_double({}), rand_double({})};
            if (rand_mat < 0.6) {
                // diffuse
                mat = make_shared<Lambertian>(std::move(albedo));
            } else if (rand_mat < 0.85) {
                // metal
                auto fuse = rand_double({{0, 0.5}});
                mat = make_shared<Metal>(std::move(albedo), fuse);
            } else {
                // glass
                mat = make_shared<Dielectric>(Vec3{0.9, 0.9, 0.9}, 1.5);
            }
            hit_list.add(make_shared<Sphere>(center, 0.2, mat));
        }
    }

    auto material1 = make_shared<Dielectric>(Vec3{1, 1, 1}, 1.5);
    hit_list.add(make_shared<Sphere>(Vec3{0, 1, 0}, 1.0, material1));

    auto material2 = make_shared<Lambertian>(Vec3{0.4, 0.2, 0.1});
    hit_list.add(make_shared<Sphere>(Vec3{-4, 1, 0}, 1.0, material2));

    auto material3 = make_shared<Metal>(Vec3{0.7, 0.6, 0.5}, 0.0);
    hit_list.add(make_shared<Sphere>(Vec3{4, 1, 0}, 1.0, material3));

    camera.render(hit_list);
}
