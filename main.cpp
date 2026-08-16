#include <iostream>
#include <math.h>
#include <memory>
#include <vector>
#include <optional>
#include <cstdlib>

#define INF 999999

const int max_depth = 10;
const int samp_size = 20;
const double samp_scale = 1.0 / samp_size;

using namespace std;

inline double linear_to_gamma(double linear_component) {
    if (linear_component > 0) {
        return sqrt(linear_component);
    }
    return 0;
}

struct Interval {
    double min;
    double max;
    bool surrounds(double target) const {
        return target > min && target < max;
    }
    double clamp(double target) const {
        if (target < min) {
            return min;
        }
        if (target > max) {
            return max;
        }
        return target;
    }
};

double rand_double(optional<Interval> i) {
    auto r = rand() / (RAND_MAX + 1.0);
    if (!i.has_value()) {
        return r;
    }
    return i->min + (i->max - i->min) * r;
}

class Vec3 {
public:
    double e[3];

    Vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}
    double x() const {
        return e[0];
    }
    double y() const {
        return e[1];
    }
    double z() const {
        return e[2];
    }

    Vec3 operator-() const {
        return {-x(), -y(), -z()};
    }

    Vec3 operator+(const Vec3 &v) const {
        Vec3 ret = *this;
        ret.e[0] += v.e[0];
        ret.e[1] += v.e[1];
        ret.e[2] += v.e[2];
        return ret;
    }
    Vec3 operator-(const Vec3 &v) const {
        Vec3 ret = *this;
        ret.e[0] -= v.e[0];
        ret.e[1] -= v.e[1];
        ret.e[2] -= v.e[2];
        return ret;
    }
    Vec3 operator*(const Vec3 &v) const {
        Vec3 ret = *this;
        ret.e[0] *= v.e[0];
        ret.e[1] *= v.e[1];
        ret.e[2] *= v.e[2];
        return ret;
    }
    Vec3 operator*(double t) const {
        Vec3 ret = *this;
        ret.e[0] *= t;
        ret.e[1] *= t;
        ret.e[2] *= t;
        return ret;
    }
    Vec3 operator/(double t) const {
        Vec3 ret = *this;
        ret.e[0] /= t;
        ret.e[1] /= t;
        ret.e[2] /= t;
        return ret;
    }

    double dot(const Vec3 &v) const {
        return e[0] * v.e[0]
            + e[1] * v.e[1]
            + e[2] * v.e[2];
    }

    double length_squared() const {
        return x() * x() + y() * y() + z() * z();
    }

    double length() const {
        return sqrt(length_squared());
    }

    static Vec3 random(optional<Interval> i) {
        return Vec3(rand_double(i), rand_double(i), rand_double(i));
    }

    bool near_zero() const {
        auto s = 1e-8;
        return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
    }
};

inline Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - n * 2 * v.dot(n);
}

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

    HitRecord(Vec3 point, Vec3 outward_normal, double t, shared_ptr<Material> mat): point(point), normal(outward_normal), t(t), mat(mat) {
        if (point.dot(outward_normal) > 0) {
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
        HitRecord rec{p, normal, t, this->mat};
        return rec;
    }
};

void write_color(Vec3 &&color) {
    auto r = linear_to_gamma(color.x());
    auto g = linear_to_gamma(color.y());
    auto b = linear_to_gamma(color.z());

    static const Interval intensity{0.000, 0.999};
    int ri = int(255.999 * intensity.clamp(r));
    int gi = int(255.999 * intensity.clamp(g));
    int bi = int(255.999 * intensity.clamp(b));
    cout << ri << " " << gi << " " << bi << endl;
}

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
        auto dir = reflect(ray.dir, rec.normal);
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

int main() {
    double ratio = 16.0 / 9.0;
    int width = 400;
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
    auto mat_left = make_shared<Metal>(Vec3(0.8, 0.8, 0.8), 0.3);
    auto mat_right = make_shared<Metal>(Vec3(0.8, 0.6, 0.2), 1);
    HittableList hit_list;
    hit_list.add(make_shared<Sphere>(Vec3{0, -100.5, -1}, 100, mat_ground));
    hit_list.add(make_shared<Sphere>(Vec3{0, 0, -1.2}, 0.5, mat_center));
    hit_list.add(make_shared<Sphere>(Vec3{-1, 0, -1}, 0.5, mat_left));
    hit_list.add(make_shared<Sphere>(Vec3{1, 0, -1}, 0.5, mat_right));

    // Render
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            Vec3 color(0, 0, 0);
            for (int _i = 0; _i < samp_size; _i++) {
                double off_x = rand_double({});
                double off_y = rand_double({});
                auto pixel_center = pixel000_loc + (pixel_delta_u * (i + off_x)) + (pixel_delta_v * (j + off_y));
                auto ray_dir = pixel_center - camera_center;
                Ray r(camera_center, ray_dir);
                color = color + calc_ray_color(r, hit_list, max_depth);
            }
            write_color(color * samp_scale);
        }
    }
}
