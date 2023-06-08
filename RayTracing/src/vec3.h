#ifndef VEC3H
#define VEC3H

#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include "glm/glm.hpp"

using glm::vec3;

class Color {
private:     
    __device__ float clamp(float value, float lower, float upper) {
        float result = value;
        if (lower > value) return lower;
        else if (upper < value) return upper;
        else return value;
    }
public:
    __device__ Color(float r, float g, float b, float a) {
        this->r = r;
        this->g = g;
        this->b = b;
        this->a = a;
    }
    float r;
    float g;
    float b;
    float a;
public: 
    __device__ uint32_t ConvertToRGBA()
    {
        uint8_t r = (uint8_t)(this->r * 255.0f);
        uint8_t g = (uint8_t)(this->g * 255.0f);
        uint8_t b = (uint8_t)(this->b * 255.0f);
        uint8_t a = (uint8_t)(this->a * 255.0f);

        uint32_t result = (a << 24) | (b << 16) | (g << 8) | r;
        return result;
    }
    __device__ Color Clamp(float min, float max) {
        Color out{0,0,0,0};
        out.r = clamp(r, min, max);
        out.g = clamp(g, min, max);
        out.b = clamp(b, min, max);
        out.a = clamp(a, min, max);
        return out; 
        
    }
private:
};

//
//class vec3 {
//
//
//public:
//    vec3() {}
//    __device__ vec3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }
//    __device__ inline float x() const { return e[0]; }
//    __device__ inline float y() const { return e[1]; }
//    __device__ inline float z() const { return e[2]; }
//    __device__ inline float r() const { return e[0]; }
//    __device__ inline float g() const { return e[1]; }
//    __device__ inline float b() const { return e[2]; }
//
//    __device__ inline const vec3& operator+() const { return *this; }
//    __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
//    __device__ inline float operator[](int i) const { return e[i]; }
//    __device__ inline float& operator[](int i) { return e[i]; };
//
//    __device__ inline vec3& operator+=(const vec3& v2);
//    __device__ inline vec3& operator-=(const vec3& v2);
//    __device__ inline vec3& operator*=(const vec3& v2);
//    __device__ inline vec3& operator/=(const vec3& v2);
//    __device__ inline vec3& operator*=(const float t);
//    __device__ inline vec3& operator/=(const float t);
//
//    __device__ inline float length() const { return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); }
//    __device__ inline float squared_length() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }
//    __device__ inline void make_unit_vector();
//
//
//    float e[3];
//};
//
//
//
//__device__ inline std::istream& operator>>(std::istream& is, vec3& t) {
//    is >> t.e[0] >> t.e[1] >> t.e[2];
//    return is;
//}
//
//__device__ inline std::ostream& operator<<(std::ostream& os, const vec3& t) {
//    os << t.e[0] << " " << t.e[1] << " " << t.e[2];
//    return os;
//}
//
//__device__ inline void vec3::make_unit_vector() {
//    float k = 1.0 / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
//    e[0] *= k; e[1] *= k; e[2] *= k;
//}
//
//__device__ inline vec3 operator+(const vec3& v1, const vec3& v2) {
//    return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
//}
//
//__device__ inline vec3 operator-(const vec3& v1, const vec3& v2) {
//    return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
//}
//
//__device__ inline vec3 operator*(const vec3& v1, const vec3& v2) {
//    return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
//}
//
//__device__ inline vec3 operator/(const vec3& v1, const vec3& v2) {
//    return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
//}
//
//__device__ inline vec3 operator*(float t, const vec3& v) {
//    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
//}
//
//__device__ inline vec3 operator/(vec3 v, float t) {
//    return vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
//}
//
//__device__ inline vec3 operator*(const vec3& v, float t) {
//    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
//}
//
//__device__ inline vec3 operator-(const vec3& v1, float t) {
//    return vec3(v1.e[0] - t, v1.e[1] - t, v1.e[2] - t);
//}
//
//__device__ inline float dot(const vec3& v1, const vec3& v2) {
//    return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
//}
//
//__device__ inline vec3 cross(const vec3& v1, const vec3& v2) {
//    return vec3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
//        (-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
//        (v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
//}
//
//
//__device__ inline vec3& vec3::operator+=(const vec3& v) {
//    e[0] += v.e[0];
//    e[1] += v.e[1];
//    e[2] += v.e[2];
//    return *this;
//}
//
//__device__ inline vec3& vec3::operator*=(const vec3& v) {
//    e[0] *= v.e[0];
//    e[1] *= v.e[1];
//    e[2] *= v.e[2];
//    return *this;
//}
//
//__device__ inline vec3& vec3::operator/=(const vec3& v) {
//    e[0] /= v.e[0];
//    e[1] /= v.e[1];
//    e[2] /= v.e[2];
//    return *this;
//}
//
//__device__ inline vec3& vec3::operator-=(const vec3& v) {
//    e[0] -= v.e[0];
//    e[1] -= v.e[1];
//    e[2] -= v.e[2];
//    return *this;
//}
//
//__device__ inline vec3& vec3::operator*=(const float t) {
//    e[0] *= t;
//    e[1] *= t;
//    e[2] *= t;
//    return *this;
//}
//
//__device__ inline vec3& vec3::operator/=(const float t) {
//    float k = 1.0 / t;
//
//    e[0] *= k;
//    e[1] *= k;
//    e[2] *= k;
//    return *this;
//}
//
//__device__ inline vec3 unit_vector(vec3 v) {
//    return v / v.length();
//}

#endif