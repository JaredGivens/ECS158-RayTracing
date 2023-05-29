#include "geo.hh"

quat_t &from_units(quat_t &that, vecf_t const &from, vecf_t const &to) {

  that.w = dot(from, to) + 1;

  if (that.w < std::numeric_limits<float>::epsilon()) {
    that.w = 0;

    if (abs(from.x) < abs(from.z)) {
      that.x = -from.y;
      that.y = from.x;
      that.z = 0;
    } else {
      that.x = 0;
      that.y = -from.z;
      that.z = from.y;
    }
  } else {

    that.x = from.y * to.z - from.z * to.y;
    that.y = from.z * to.x - from.x * to.z;
    that.z = from.x * to.y - from.y * to.x;
  }

  return normalize(that);
}

quat_t &zero(quat_t &that) {
  that.x = 0;
  that.y = 0;
  that.z = 0;
  that.w = 1;
  return that;
}

quat_t &normalize(quat_t &that) {
  float const l = len(that);
  if (l == 0) {
    that.w = 1;
  } else {
    that.x /= l;
    that.y /= l;
    that.z /= l;
    that.w /= l;
  }
  return that;
}

float len(quat_t const &that) {
  return sqrt(that.x * that.x + that.y * that.y + that.z * that.z +
              that.w * that.w);
}

quat_t &mul(quat_t &that, quat_t const &q) {
  float const x = that.x, y = that.y, z = that.z, w = that.w;

  that.x = x * q.w + w * q.x + y * q.z - z * q.y;
  that.y = y * q.w + w * q.y + z * q.x - x * q.z;
  that.z = z * q.w + w * q.z + x * q.y - y * q.x;
  that.w = w * q.w - x * q.x - y * q.y - z * q.z;

  return that;
}

quat_t &pre_mul(quat_t &that, quat_t const &q) {
  float const x = that.x, y = that.y, z = that.z, w = that.w;

  that.x = q.x * w + q.w * x + q.y * z - q.z * y;
  that.y = q.y * w + q.w * y + q.z * x - q.x * z;
  that.z = q.z * w + q.w * z + q.x * y - q.y * x;
  that.w = q.w * w - q.x * x - q.y * y - q.z * z;

  return that;
}

// assumes axis is normalized
quat_t &from_axis(quat_t &that, vecf_t const &axis, float ang) {
  float const half = ang / 2, s = sin(half);

  that.x = axis.x * s;
  that.y = axis.y * s;
  that.z = axis.z * s;
  that.w = cos(half);

  return that;
}

float constexpr kEpsilon = std::numeric_limits<float>::epsilon();
bool near_zero(vecf_t const & that) {
  return abs(that.x) <= kEpsilon && abs(that.y) <= kEpsilon && abs(that.z) <= kEpsilon;
}

tsf_t &copy(tsf_t &that, tsf_t const &t) {
  that.pos = t.pos;
  that.rot = t.rot;
  return that;
}

std::ostream &operator<<(std::ostream &os, quat_t const &q) {
  os << '(' << q.x << ", " << q.y << ", " << q.z << ", " << q.w << ')';
  return os;
}

vecf_t &compute_mid(vecf_t &v, tri_t const &t) {
  return div(add(add(v = t.a, t.b), t.c), 3);
}

vecf_t &compute_norm(vecf_t &v, tri_t const &t) {
  vecf_t v0 = t.c;
  return normalize(cross(sub(v = t.b, t.a), sub(v0, t.a)));
}

int32_t closest(vecf_t const &v, tri_t const &t) {
  vecf_t v0;
  int32_t rec = 0;
  float rec_diff = len_sq(sub(v0 = v, t.a));
  for (int32_t i = 1; i < 3; ++i) {
    float diff = len_sq(sub(v0 = v, t.vecs[i]));
    if (diff < rec_diff) {
      rec_diff = diff;
      rec = i;
    }
  }

  return rec;
}

std::ostream &operator<<(std::ostream &os, tri_t const &that) {
  os << '[' << that.a << "; " << that.b << "; " << that.c << ']';
  return os;
}

vecf_t compute_bary(tri_t const &tri, vecf_t const &vec) {
  vecf_t v0, v1, v2;
  sub(v0 = tri.c, tri.a);
  sub(v1 = tri.b, tri.a);
  sub(v2 = vec, tri.a);

  float const dot00 = dot(v0, v0);
  float const dot01 = dot(v0, v1);
  float const dot02 = dot(v0, v2);
  float const dot11 = dot(v1, v1);
  float const dot12 = dot(v1, v2);

  float const denom = dot00 * dot11 - dot01 * dot01;

  vecf_t res{-2, -1, -1};

  // not colinear or singular triangle
  if (denom != 0) {

    float const invDenom = 1 / denom;
    float const u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    float const v = (dot00 * dot12 - dot01 * dot02) * invDenom;
    set(res, 1 - u - v, v, u);
  }

  // barycentric coordinates must always sum to 1
  return res;
}
bool contains_point(tri_t const &tri, vecf_t const &v) {
  vecf_t v0 = compute_bary(tri, v);
  return (v0.x >= 0) && (v0.y >= 0) && ((v0.x + v0.y) <= 1);
}

bool Ray::intersect(vecf_t &target, tri_t const &t, bool cull_back) const {

  vecf_t e1 = t.b;
  vecf_t e2 = t.c;
  sub(e1, t.a);
  sub(e2, t.a);

  vecf_t n = e1;
  cross(n, e2);

  float DdN = dot(dir_, n);
  int32_t sign;

  if (DdN > 0) {

    if (cull_back) {
      return false;
    }
    sign = 1;

  } else if (DdN < 0) {

    sign = -1;
    DdN = -DdN;

  } else {

    return false;
  }

  vecf_t diff = origin_;
  sub(diff, t.a);
  float DdQxE2 = sign * dot(dir_, pre_cross(e2, diff));

  // b1 < 0, no intersection
  if (DdQxE2 < 0) {

    return false;
  }

  float DdE1xQ = sign * dot(dir_, cross(e1, diff));

  // b2 < 0, no intersection
  if (DdE1xQ < 0) {

    return false;
  }

  // b1+b2 > 1, no intersection
  if (DdQxE2 + DdE1xQ > DdN) {

    return false;
  }

  // Line intersects triangle, check if ray does
  float QdN = -sign * dot(diff, n);

  // t < 0, no intersection
  if (QdN < 0) {

    return false;
  }

  target = at(QdN / DdN);
  return true;
}

vecf_t Ray::at(float mag) const {
  vecf_t res = dir_;
  return add(mul(res = dir_, mag), origin_);
}

bool Ray::intersect(Intersect &intersect, Sphere const &sphere, float t_min,
                    float t_max) const {
   vecf_t oc = origin_;
   sub(oc, sphere.center);
    auto half_b = dot(oc, dir_);
    auto c = len_sq(oc) - sphere.radius*sphere.radius;

    auto discriminant = half_b*half_b - c;
    if (discriminant < 0) {
      return false;
    }
    auto sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    auto root = (-half_b - sqrtd);
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd);
        if (root < t_min || t_max < root) {
            return false;
        }
    }

  intersect.dist = root;
  intersect.pos = at(intersect.dist);
  div(sub(intersect.normal = intersect.pos, sphere.center), sphere.radius);
  intersect.front = dot(dir_, intersect.normal) < 0;
  if(!intersect.front) {
    mul(intersect.normal, -1);
  }
  intersect.mat = sphere.mat;
  return true;
}