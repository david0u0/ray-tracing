#ifndef CAMERA_H
#define CAMERA_H

#include "common.cuh"
#include "world.cuh"

#define INF 999999

struct CamConfig {
    int width;
    double ratio;
    double virtical_fov = 3.1415926535 / 2; // field of view;
    Vec3 look_at = {0, 0, -1};
    Vec3 vup = {0, 1, 0};
    Vec3 camera_center = {0, 0, 0};
    int max_depth = 20;
    int samp_size = 20;
};

class Camera {
private:
    Vec3 pixel00_loc;
    Vec3 pixel_delta_u;
    Vec3 pixel_delta_v;
    int max_depth;
    int samp_size;
    double samp_scale;
    Vec3 camera_center = {0, 0, 0};

public:
    int width;
    int height;

    Camera(CamConfig conf): width(conf.width), camera_center(conf.camera_center) {
        samp_size = conf.samp_size;
        samp_scale = 1.0 / samp_size;
        max_depth = conf.max_depth;
        height = int(width / conf.ratio);

        auto w = camera_center - conf.look_at;
        double focal_length = w.length();
        w = w / focal_length;
        auto u = conf.vup.cross(w).to_unit();
        auto v = w.cross(u);

        auto h = tan(conf.virtical_fov/2);
        double viewport_height = 2.0 * h * focal_length;
        double viewport_width = viewport_height * (double(width) / height);
        auto viewport_u = u * viewport_width;
        auto viewport_v = -v * viewport_height;

        pixel_delta_u = viewport_u / width;
        pixel_delta_v = viewport_v / height;

        auto viewport_upper_left = conf.look_at - viewport_u/2 - viewport_v/2;
        pixel00_loc = viewport_upper_left + (pixel_delta_u + pixel_delta_v) * 0.5;
    }

    void render(const HittableList &world) {
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                Vec3 color(0, 0, 0);
                for (int _i = 0; _i < samp_size; _i++) {
                    double off_x = rand_double({}) - 0.5;
                    double off_y = rand_double({}) - 0.5;
                    auto pixel_center = pixel00_loc + (pixel_delta_u * (i + off_x)) + (pixel_delta_v * (j + off_y));
                    auto ray_dir = pixel_center - camera_center;
                    Ray r(camera_center, ray_dir);
                    auto cur_col = calc_ray_color(r, world, max_depth);
                    color = color + cur_col;
                }
                write_color(color * samp_scale);
            }
        }
    }

    Vec3 calc_ray_color(const Ray &r, const Hittable &world, int depth) {
        if (depth == 0) {
            return {0, 0, 0};
        }

        auto rec = world.hit(r, {0.001, INF}); // Fix shadow acne by ignoring too small a time
        if (rec.has_value()) {
            auto scatter_rec = rec->mat->scatter(r, *rec);
            if (!scatter_rec.has_value()) {
                return {0, 0, 0};
            }
            return calc_ray_color(scatter_rec->ray, world, depth - 1) * scatter_rec->attenuation;
        }

        Vec3 dir = r.dir;
        Vec3 unit = dir / dir.length();
        auto a = 0.5 * (unit.y() + 1.0);
        return Vec3{1.0, 1.0, 1.0} * (1.0-a) + Vec3{0.5, 0.7, 1.0} * a;
    }
};

#endif
