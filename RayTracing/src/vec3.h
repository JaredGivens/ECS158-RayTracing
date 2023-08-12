#ifndef VEC3H
#define VEC3H

#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include "glm/glm.hpp"

using glm::vec3;
using glm::vec4;

class Color {
private:     
    __device__ float clamp(float value, float lower, float upper) {
        if (lower > value) return lower;
        else if (upper < value) return upper;
        else return value;
    }
public:
    __device__ Color(vec4 rgba) {
        r = rgba.r;
        g = rgba.g;
        b = rgba.b;
        a = rgba.a;
    }
    __device__ Color(float r, float g, float b, float a) {
        this->r = r;
        this->g = g;
        this->b = b;
        this->a = a;
    }
    __device__ Color(float r, float g, float b, float a, bool _unused) {
        this->r = r;
        this->g = g;
        this->b = b;
        this->a = a;
    }
    __device__ vec4 to_vec4() {
        return vec4{ r,g,b,a };
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

#endif