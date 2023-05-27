#ifndef DRYF_GEO_HH
#define DRYF_GEO_HH

#include "geo.h"
#include <concepts>
#include <iostream>

vecf_t const kUnitX{1, 0, 0};
vecf_t const kUnitY{0, 1, 0};
vecf_t const kUnitZ{0, 0, 1};

float const kHalfPi = acos(0);
float const kPi = kHalfPi * 2;
float const kTau = kPi * 2;

template <typename T>
concept Arithmatic = std::is_arithmetic<T>::value;

template <typename V>
concept Vec3_like = requires {
  Arithmatic<decltype(V::x)>;
  Arithmatic<decltype(V::y)>;
  Arithmatic<decltype(V::z)>;
};

template <typename T> T sign(T val) { return (T(0) < val) - (val < T(0)); }

std::ostream &operator<<(std::ostream &os, quat_t const &q);
quat_t &from_units(quat_t &, vecf_t const &, vecf_t const &);
quat_t &normalize(quat_t &);
quat_t &zero(quat_t &);
float len(quat_t const &);
quat_t &mul(quat_t &, quat_t const &);
quat_t &pre_mul(quat_t &, quat_t const &);
quat_t &from_axis(quat_t &, vecf_t const &, float);
bool near_zero(vecf_t const &);

template <Vec3_like V, Arithmatic A> V &set(V &that, A x, A y, A z) {
  that.x = x;
  that.y = y;
  that.z = z;
  return that;
}

template <Vec3_like V> V &zero(V &that) {
  that.x = 0;
  that.y = 0;
  that.z = 0;
  return that;
}

template <Vec3_like V> bool is_zero(V const &that) {
  return that.x == 0 && that.y == 0 && that.z == 0;
}

template <Vec3_like V> V &add(V &that, V const &v) {
  that.x += v.x;
  that.y += v.y;
  that.z += v.z;
  return that;
}

template <Vec3_like V> V &sub(V &that, V const &v) {
  that.x -= v.x;
  that.y -= v.y;
  that.z -= v.z;
  return that;
}

template <Vec3_like V> V &mul(V &that, V const &v) {
  that.x *= v.x;
  that.y *= v.y;
  that.z *= v.z;
  return that;
}

template <Vec3_like V> V &div(V &that, V const &v) {
  that.x /= v.x;
  that.y /= v.y;
  that.z /= v.z;
  return that;
}

template <Vec3_like V, Arithmatic A> V &add(V &that, A s) {
  that.x += s;
  that.y += s;
  that.z += s;
  return that;
}

template <Vec3_like V, Arithmatic A> V &sub(V &that, A s) {
  that.x -= s;
  that.y -= s;
  that.z -= s;
  return that;
}

template <Vec3_like V, Arithmatic A> V &div(V &that, A s) {
  that.x /= s;
  that.y /= s;
  that.z /= s;
  return that;
}

template <Vec3_like V, Arithmatic A> V &mul(V &that, A s) {
  that.x *= s;
  that.y *= s;
  that.z *= s;
  return that;
}

template <Vec3_like V> V &lerp(V &that, V const &v, float a) {
  that.x += (v.x - that.x) * a;
  that.y += (v.y - that.y) * a;
  that.z += (v.z - that.z) * a;
  return that;
}

template <Vec3_like V, Arithmatic A> V &add_len(V &that, A s) {
  A l = len(that);
  that.x += (that.x / l) * s;
  that.y += (that.y / l) * s;
  that.z += (that.z / l) * s;
  return that;
}

template <Vec3_like V> decltype(V::x) dot(V const &that, V const &v) {
  return that.x * v.x + that.y * v.y + that.z * v.z;
}

template <Vec3_like V> decltype(V::x) len_sq(V const &that) {
  return that.x * that.x + that.y * that.y + that.z * that.z;
}

template <Vec3_like V> decltype(V::x) len(V const &that) {
  return sqrt(that.x * that.x + that.y * that.y + that.z * that.z);
}

template <Vec3_like V> V &pre_cross(V &that, V const &v) {

  decltype(V::x) const x = that.x, y = that.y, z = that.z;

  that.x = v.y * z - v.z * y;
  that.y = v.z * x - v.x * z;
  that.z = v.x * y - v.y * x;

  return that;
}

template <Vec3_like V> V &reflect(V &that, V const &n) {
  vecf_t v0 = n;
  return sub(that, mul(v0, 2 * dot(that, n)));
}

template <Vec3_like V> V &cross(V &that, V const &v) {

  decltype(V::x) const x = that.x, y = that.y, z = that.z;

  that.x = y * v.z - z * v.y;
  that.y = z * v.x - x * v.z;
  that.z = x * v.y - y * v.x;

  return that;
}

template <Vec3_like V> V &mul(V &that, quat_t const &q) {
  float const ix = q.w * that.x + q.y * that.z - q.z * that.y;
  float const iy = q.w * that.y + q.z * that.x - q.x * that.z;
  float const iz = q.w * that.z + q.x * that.y - q.y * that.x;
  float const iw = -q.x * that.x - q.y * that.y - q.z * that.z;

  that.x = ix * q.w + iw * -q.x + iy * -q.z - iz * -q.y;
  that.y = iy * q.w + iw * -q.y + iz * -q.x - ix * -q.z;
  that.z = iz * q.w + iw * -q.z + ix * -q.y - iy * -q.x;

  return that;
}

template <Vec3_like V> V &abs(V &that) {
  that.x = std::abs(that.x);
  that.y = std::abs(that.y);
  that.z = std::abs(that.z);
  return that;
}

template <Vec3_like V> V &normalize(V &that) { return div(that, len(that)); }

template <Vec3_like V>
std::ostream &operator<<(std::ostream &os, V const &that) {
  os << '(' << that.x << ", " << that.y << ", " << that.z << ')';
  return os;
}

template <Vec3_like V, Arithmatic A> V &add(V &that, V const &v, A s) {
  that.x += v.x * s;
  that.y += v.y * s;
  that.z += v.z * s;
  return that;
}

// assumes norm normalized
template <Vec3_like V> V &project(V &that, V const &norm) {
  V v0;
  sub(that, mul(v0 = norm, dot(that, norm)));
  return that;
}

template <class T> inline void hash_combine(int32_t &seed, const T &v) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <Vec3_like V> int32_t hash(V const &that) {
  int32_t s = 0;
  hash_combine(s, that.x);
  hash_combine(s, that.y);
  hash_combine(s, that.z);
  return s;
}

// seg_t &set(seg_t &, vecf_t const &, vecf_t const &);
vecf_t &compute_norm(vecf_t &, tri_t const &);
vecf_t &compute_mid(vecf_t &, tri_t const &);

vecf_t compute_bary(tri_t const &, vecf_t const &);

bool contains_point(tri_t const &, vecf_t const &);

// vert index of vert closest to v in t
int32_t closest(vecf_t const &, tri_t const &);

template <typename I>
bool sat_for_axes(I begin, I end, vecf_t const &v0, vecf_t const &v1,
                  vecf_t const &v2, vecf_t const &extents) {

  return std::all_of(begin, end, [&](vecf_t const &axe) {
    auto const r = extents.x * std::abs(axe.x) + extents.y * std::abs(axe.y) +
                   extents.z * std::abs(axe.z);
    auto const p0 = dot(v0, axe);
    auto const p1 = dot(v1, axe);
    auto const p2 = dot(v2, axe);
    return std::max(-std::max({p0, p1, p2}), std::min({p0, p1, p2})) <= r;
  });
}

struct Material {
  vecf_t color;
  float metalness;
  float specular;
  float roughness;
};

struct Sphere {
  vecf_t center;
  float radius;
  Material mat;
};

struct Intersect {
  Material mat;
  vecf_t pos;
  vecf_t normal;
  float dist;
  bool front;
};

class Ray {
public:
  vecf_t origin_;
  vecf_t dir_;
  bool intersect(vecf_t &target, tri_t const &t, bool cull_back) const;
  bool intersect(Intersect &, Sphere const &, float t_min, float t_max) const;
  vecf_t at(float mag) const;
};

#endif