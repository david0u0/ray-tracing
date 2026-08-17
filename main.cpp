#include "common.cuh"

#include <iostream>
#include <math.h>
#include <memory>
#include <vector>
#include <optional>
#include <cstdlib>


#define INF 999999

const int max_depth = 20;
const int samp_size = 20;
const double samp_scale = 1.0 / samp_size;

class Ray {
public:
    Vec3 orig;
    Vec3 dir;

    Ray(const Vec3 &orig, const Vec3 &dir): orig(orig), dir(dir) {}
    Vec3 at(double t) const {
        return orig + dir * t;
    }
};

struct ScatterRecord {
    Vec3 attenuation;
    Ray ray;
};

class HitRecord;
class Material {
public:
    virtual ~Material() = default;
    virtual optional<ScatterRecord> scatter(const Ray &ray, const HitRecord &rec) const {
        return {};
    }
};

struct HitRecord {
    Vec3 point;
    Vec3 normal;
    double t;
    bool front_face;
    shared_ptr<Material> mat;

    HitRecord(const Vec3 &ray_dir, Vec3 point, Vec3 outward_normal, double t, shared_ptr<Material> mat): point(point), normal(outward_normal), t(t), mat(mat) {
        if (ray_dir.dot(outward_normal) > 0) {
            this->front_face = false;
            this->normal = -this->normal;
        } else {
            this->front_face = true;
        }
    }
};

class Hittable {
public:
    virtual ~Hittable() = default;
    virtual optional<HitRecord> hit(const Ray &ray, Interval t_interval) const = 0;
};

class HittableList: public Hittable {
    vector<shared_ptr<Hittable>> list;
public:
    void add(shared_ptr<Hittable> ptr) {
        this->list.push_back(ptr);

    }
    optional<HitRecord> hit(const Ray &ray, Interval t_interval) const override {
        optional<HitRecord> ret = {};
        for (auto &obj : this->list) {
            auto rec = obj->hit(ray, t_interval);
            if (!rec.has_value()) {
                continue;
            }

            t_interval.max = rec->t;
            ret = rec;
        }
        return ret;
    }
};

class Sphere: public Hittable {
    Vec3 center;
    double radius;
    shared_ptr<Material> mat;
public:
    Sphere(Vec3 center, double radius, shared_ptr<Material> mat): center(center), radius(radius), mat(mat) {}
    virtual optional<HitRecord> hit(const Ray &ray, Interval t_interval) const override {
        Vec3 oc = center - ray.orig;
        auto a = ray.dir.dot(ray.dir);
        /*auto b = ray.dir.dot(oc) * -2;*/
        auto h = ray.dir.dot(oc);
        auto c = oc.dot(oc) - radius * radius;
        auto descr = h * h - a * c;
        if (descr < 0.0) {
            return {};
        }

        double root = sqrt(descr);
        double t = (h - root) / a;
        if (!t_interval.surrounds(t)) {
            t = (h + root) / a;
            if (!t_interval.surrounds(t)) {
                return {};
            }
        }

        auto p = ray.at(t);
        auto normal = (p - this->center) / this->radius;
        HitRecord rec{ray.dir, p, normal, t, this->mat};
        return rec;
    }
};

inline Vec3 random_unit_vector() {
    while (true) {
        auto p = Vec3::random({{-1, 1}});
        auto lensq = p.length_squared();
        if (lensq <= 1 && lensq > 1e-160) {
            // QUESTION: why not just always return? What's wrong with having a distribution of cube instead of sphere?
            return p / sqrt(lensq);
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

class Metal : public Material {
public:
    Vec3 albedo;
    double fuse;
    Metal(Vec3 &&albedo, double fuse): albedo(albedo), fuse(fuse) {}
    optional<ScatterRecord> scatter(const Ray &ray, const HitRecord &rec) const override {
        auto fuse_v = random_unit_vector() * this->fuse;
        auto dir = rec.normal.reflect(ray.dir);
        dir = dir / dir.length() + fuse_v;
        if (dir.dot(rec.normal) > 0) {
            return {{this->albedo, Ray(rec.point, dir)}};
        }
        return {};
    }
};
class Lambertian : public Material {
public:
    Vec3 albedo;
    Lambertian(Vec3 &&albedo): albedo(albedo) {}
    optional<ScatterRecord> scatter(const Ray &ray, const HitRecord &rec) const override {
        auto dir = rec.normal + random_unit_vector();
        if (dir.near_zero()) {
            dir = rec.normal;
        }
        return {{this->albedo, Ray(rec.point, dir)}};
    }
};
class Dielectric : public Material {
public:
    Vec3 albedo;
    double refraction_index;
    double refraction_index_inverse;
    Dielectric(Vec3 &&albedo, double refraction_index): albedo(albedo), refraction_index(refraction_index) {
        refraction_index_inverse = 1.0 / refraction_index;
    }

    optional<ScatterRecord> scatter(const Ray &ray, const HitRecord &rec) const override {
        double ri = rec.front_face ? refraction_index_inverse : refraction_index;

        auto v = ray.dir.to_unit();
        auto cos_theta = fmin(-rec.normal.dot(v), 1.0);
        auto sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0;
        if (cannot_refract || reflectance(cos_theta, ri) > rand_double({})) {
            auto dir = rec.normal.reflect(v);
            return {{this->albedo, Ray(rec.point, dir)}};
        }

        auto dir = rec.normal.refract(v, ri);
        return {{this->albedo, Ray(rec.point, dir)}};
    }
private:
    static double reflectance(double cosine, double refraction_index) {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1 - refraction_index) / (1 + refraction_index);
        r0 = r0*r0;
        return r0 + (1-r0)*std::pow((1 - cosine),5);
    }
};

int main() {
    double ratio = 16.0 / 9.0;
    int width = 1200;
    int height = int(width / ratio);


    cout << "P3" << endl;
    cout << width << " " << height << endl;
    cout << 255 << endl;

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
    hit_list.add(make_shared<Sphere>(Vec3{-1, 0, -1}, 0.35, mat_bubble));
    hit_list.add(make_shared<Sphere>(Vec3{1, 0, -1}, 0.5, mat_right));

    // Render
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            Vec3 color(0, 0, 0);
            for (int _i = 0; _i < samp_size; _i++) {
                double off_x = rand_double({}) - 0.5;
                double off_y = rand_double({}) - 0.5;
                auto pixel_center = pixel000_loc + (pixel_delta_u * (i + off_x)) + (pixel_delta_v * (j + off_y));
                auto ray_dir = pixel_center - camera_center;
                Ray r(camera_center, ray_dir);
                auto cur_col = calc_ray_color(r, hit_list, max_depth);
                color = color + cur_col;
            }
            write_color(color * samp_scale);
        }
    }
}
