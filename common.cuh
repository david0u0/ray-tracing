#include <math.h>
#include <cstdlib>
#include <iostream>
#include <optional>

using namespace std;

#ifdef CUDA
#define CUDA_FUNC() __host__ __device__                                                 
#else
#define CUDA_FUNC()
#endif

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

inline double linear_to_gamma(double linear_component) {
    if (linear_component > 0) {
        return sqrt(linear_component);
    }
    return 0;
}

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

    CUDA_FUNC() Vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}
    CUDA_FUNC() double x() const {
        return e[0];
    }
    CUDA_FUNC() double y() const {
        return e[1];
    }
    CUDA_FUNC() double z() const {
        return e[2];
    }

    CUDA_FUNC() Vec3 operator-() const {
        return {-x(), -y(), -z()};
    }

    CUDA_FUNC() Vec3 operator+(const Vec3 &v) const {
        Vec3 ret = *this;
        ret.e[0] += v.e[0];
        ret.e[1] += v.e[1];
        ret.e[2] += v.e[2];
        return ret;
    }
    CUDA_FUNC() Vec3 operator-(const Vec3 &v) const {
        Vec3 ret = *this;
        ret.e[0] -= v.e[0];
        ret.e[1] -= v.e[1];
        ret.e[2] -= v.e[2];
        return ret;
    }
    CUDA_FUNC() Vec3 operator*(const Vec3 &v) const {
        Vec3 ret = *this;
        ret.e[0] *= v.e[0];
        ret.e[1] *= v.e[1];
        ret.e[2] *= v.e[2];
        return ret;
    }
    CUDA_FUNC() Vec3 operator*(double t) const {
        Vec3 ret = *this;
        ret.e[0] *= t;
        ret.e[1] *= t;
        ret.e[2] *= t;
        return ret;
    }
    CUDA_FUNC() Vec3 operator/(double t) const {
        Vec3 ret = *this;
        ret.e[0] /= t;
        ret.e[1] /= t;
        ret.e[2] /= t;
        return ret;
    }

    CUDA_FUNC() double dot(const Vec3 &v) const {
        return e[0] * v.e[0]
            + e[1] * v.e[1]
            + e[2] * v.e[2];
    }

    CUDA_FUNC() double length_squared() const {
        return x() * x() + y() * y() + z() * z();
    }

    CUDA_FUNC() double length() const {
        return sqrt(length_squared());
    }

    CUDA_FUNC() static Vec3 random(optional<Interval> i) {
        return Vec3(rand_double(i), rand_double(i), rand_double(i));
    }

    CUDA_FUNC() bool near_zero() const {
        auto s = 1e-8;
        return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
    }

    CUDA_FUNC() Vec3 to_unit() const {
        return *this / this->length();
    }
    // `this` as norm
    CUDA_FUNC() inline Vec3 reflect(const Vec3& v) const {
        auto &n = *this;
        return v - n * 2 * v.dot(n);
    }

    // `this` as unit norm, and v as unit vector
    CUDA_FUNC() inline Vec3 refract(const Vec3& v, double eta_ratio) const {
        auto &n = *this;
        auto cos_theta = fmin(-n.dot(v), 1.0);
        auto sin_theta = sqrt(1.0 - cos_theta * cos_theta);
        auto perp = (v + n * cos_theta) * eta_ratio;
        auto para = n * (-sqrt(fabs(1.0 - perp.length_squared())));
        return perp + para;
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

