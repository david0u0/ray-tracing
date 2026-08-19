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

    double focus_dist = 1.0;
    double defocus_angle = 0;
};

class Camera {
private:
    Vec3 pixel00_loc;
    Vec3 pixel_delta_u;
    Vec3 pixel_delta_v;
    Vec3 defocus_disk_u;
    Vec3 defocus_disk_v;
    int max_depth;
    int samp_size;
    double samp_scale;
    Vec3 camera_center = {0, 0, 0};
    bool has_defocus = 0;

public:
    int width;
    int height;

    GPU_FUNC() void debug() {
        printf("%d %d\n", width, height);
        printf("%f %f %f\n", pixel00_loc.x(), pixel00_loc.y(), pixel00_loc.z());
        printf("%f %f %f\n", pixel_delta_u.x(), pixel_delta_u.y(), pixel_delta_u.z());
        printf("%f %f %f\n", pixel_delta_v.x(), pixel_delta_v.y(), pixel_delta_v.z());
    }

    GPU_FUNC() Camera(CamConfig conf): width(conf.width), camera_center(conf.camera_center) {
        samp_size = conf.samp_size;
        samp_scale = 1.0 / samp_size;
        max_depth = conf.max_depth;
        height = int(width / conf.ratio);

        auto w = (camera_center - conf.look_at).to_unit();
        auto u = conf.vup.cross(w).to_unit();
        auto v = w.cross(u);

        auto h = tan(conf.virtical_fov/2);
        double viewport_height = 2.0 * h * conf.focus_dist;
        double viewport_width = viewport_height * (double(width) / height);
        auto viewport_u = u * viewport_width;
        auto viewport_v = -v * viewport_height;

        pixel_delta_u = viewport_u / width;
        pixel_delta_v = viewport_v / height;

        auto viewport_upper_left = camera_center - conf.focus_dist * w - viewport_u/2 - viewport_v/2;
        pixel00_loc = viewport_upper_left + (pixel_delta_u + pixel_delta_v) * 0.5;

        has_defocus = conf.defocus_angle > 0;
        if (has_defocus) {
            auto defocus_radius = conf.focus_dist * tan(conf.defocus_angle / 2);
            defocus_disk_u = u * defocus_radius;
            defocus_disk_v = v * defocus_radius;
        }
    }

    GPU_FUNC() Vec3 render(const HittableList &world, int x, int y, curandState *st) const {
        Vec3 color(0, 0, 0);
        auto local_st = *st;
        for (int _i = 0; _i < samp_size; _i++) {
            double off_x = rand_double({}, &local_st) - 0.5;
            double off_y = rand_double({}, &local_st) - 0.5;
            auto origin = camera_center;
            if (has_defocus) {
                auto t = random_in_unit_disk(&local_st);
                origin = origin + defocus_disk_u * t.x() + defocus_disk_v * t.y();
            }
            auto pixel_center = pixel00_loc + (pixel_delta_u * (x + off_x)) + (pixel_delta_v * (y + off_y));
            auto ray_dir = pixel_center - origin;
            Ray r(origin, ray_dir);
            auto cur_col = calc_ray_color(r, world, st);
            color = color + cur_col;
        }
        return color * samp_scale;
    }

    GPU_FUNC() Vec3 calc_ray_color(Ray cur_ray, const HittableList &world, curandState *st) const {
        Vec3 cur_color = {1, 1, 1};

        for (int i = 0; i < max_depth; i++) {
            auto rec = world.hit(cur_ray, {0.001, INF});
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
};

#endif
