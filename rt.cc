#include <iostream>
#include <stdint.h>
#include <vector>
#include "geo.hh"

void write_color(std::ostream &out, vecf_t pixel_color) {
    // Write the translated [0,255] value of each color component.
    out << int32_t(255.999 * pixel_color.x) << ' '
        << int32_t(255.999 * pixel_color.y) << ' '
        << int32_t(255.999 * pixel_color.z) << '\n';
}

vecf_t ray_color(Ray const& ray, std::vector<Sphere> const &spheres) {
    Intersect inter;
    for(auto const &sphere : spheres) {
        if(ray.intersect(inter, sphere, 0.01, 100)){
            return div(add(inter.normal, 1), 2);
        }
    }

    auto t = (ray.dir_.y + 1) / 2;
    auto white = vecf_t{1, 1, 1};
    auto blue = vecf_t{0.5, 0.7, 1};
    return add(mul(white, (1.0 - t)), mul(blue, t));
}

int32_t main() {

    auto spheres = std::vector<Sphere>();
    spheres.push_back(Sphere {
        vecf_t{0, 0, -1},
        0.5
    });

    // Image

    const float aspect_ratio = 16.0 / 9.0;
    const int32_t image_height = 255;
    const int32_t image_width = image_height * aspect_ratio;

    float viewport_height = 2.0;
    float viewport_width = aspect_ratio * viewport_height;
    float focal_length = 1.0;

    vecf_t v0;
    vecf_t v1;

    auto const origin = vecf_t{0, 0, 0};
    auto const horizontal = vecf_t{viewport_width, 0, 0};
    auto const vertical = vecf_t{0, viewport_height, 0};
    auto const lower_left_corner = sub(v1 = origin, add(div(add(v0 = horizontal, vertical), 2), vecf_t{0, 0, focal_length}));

    // Render

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int32_t j = image_height - 1; j > -1; --j) {
        for (int32_t i = 0; i < image_width; ++i) {
            auto u = float(i) / float(image_width-1);
            auto v = float(j) / float(image_height-1);
            // ray from origin to pixel

            auto r = Ray(origin, normalize(sub(add(add(mul(v0 = horizontal, u), mul(v1 = vertical, v)), lower_left_corner), origin)));
            // gradient 
            vecf_t pixel_color = ray_color(r, spheres);
            write_color(std::cout, pixel_color);
        }
    }
}