#ifndef WORLD_H
#define WORLD_H

#include "common.cuh"

#include <vector>
#include <memory>

class Ray {
public:
    Vec3 orig;
    Vec3 dir;

    CUDA_FUNC() Ray(const Vec3 &orig, const Vec3 &dir): orig(orig), dir(dir) {}
    CUDA_FUNC() Vec3 at(double t) const {
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
    GPU_FUNC() virtual my_optional<ScatterRecord> scatter(const Ray &ray, const HitRecord &rec, curandState *st) const {
        return {};
    }
};

struct HitRecord {
    Vec3 point;
    Vec3 normal;
    double t;
    bool front_face;
    Material *mat;

    CUDA_FUNC() HitRecord(const Vec3 &ray_dir, Vec3 point, Vec3 outward_normal, double t, Material *mat): point(point), normal(outward_normal), t(t), mat(mat) {
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
    CUDA_FUNC() virtual my_optional<HitRecord> hit(const Ray &ray, Interval t_interval) const = 0;
};

class HittableList {
public:
    Hittable **list; // TODO: delete
    int tail;

    CUDA_FUNC() HittableList(int size) {
        this->list = new Hittable*[size];
        this->tail = 0;
    }
    CUDA_FUNC() void add(Hittable* ptr) {
        this->list[this->tail++] = ptr;

    }
    CUDA_FUNC() my_optional<HitRecord> hit(const Ray &ray, Interval t_interval) const {
        my_optional<HitRecord> ret = {};
        for (int i = 0; i < tail; i++) {
            auto &obj = list[i];
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
    Material *mat; // TODO: delete
public:
    CUDA_FUNC() Sphere(Vec3 center, double radius, Material *mat): center(center), radius(radius), mat(mat) {}
    CUDA_FUNC() virtual my_optional<HitRecord> hit(const Ray &ray, Interval t_interval) const override {
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

class Metal : public Material {
public:
    Vec3 albedo;
    double fuse;
    CUDA_FUNC() Metal(Vec3 &&albedo, double fuse): albedo(albedo), fuse(fuse) {}
    GPU_FUNC() my_optional<ScatterRecord> scatter(const Ray &ray, const HitRecord &rec, curandState *st) const override {
        auto fuse_v = random_unit_vector(st) * this->fuse;
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
    CUDA_FUNC() Lambertian(Vec3 &&albedo): albedo(albedo) {}
    GPU_FUNC() my_optional<ScatterRecord> scatter(const Ray &ray, const HitRecord &rec, curandState *st) const override {
        auto dir = rec.normal + random_unit_vector(st);
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
    GPU_FUNC() Dielectric(Vec3 &&albedo, double refraction_index): albedo(albedo), refraction_index(refraction_index) {
        refraction_index_inverse = 1.0 / refraction_index;
    }

    GPU_FUNC() my_optional<ScatterRecord> scatter(const Ray &ray, const HitRecord &rec, curandState *st) const override {
        double ri = rec.front_face ? refraction_index_inverse : refraction_index;

        auto v = ray.dir.to_unit();
        auto cos_theta = fmin(-rec.normal.dot(v), 1.0);
        auto sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0;
        if (cannot_refract || reflectance(cos_theta, ri) > rand_double({}, st)) {
            auto dir = rec.normal.reflect(v);
            return {{this->albedo, Ray(rec.point, dir)}};
        }

        auto dir = rec.normal.refract(v, ri);
        return {{this->albedo, Ray(rec.point, dir)}};
    }
private:
    GPU_FUNC() static double reflectance(double cosine, double refraction_index) {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1 - refraction_index) / (1 + refraction_index);
        r0 = r0*r0;
        return r0 + (1-r0)*std::pow((1 - cosine),5);
    }
};

#endif
