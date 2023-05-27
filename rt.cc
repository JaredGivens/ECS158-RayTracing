#include "geo.hh"
#include <iostream>
#include <stdint.h>
#include <vector>

void write_color(std::ostream &out, vecf_t pixel_color) {
  // Write the translated [0,255] value of each color component.
  out << int32_t(255.999 * pixel_color.x) << ' '
      << int32_t(255.999 * pixel_color.y) << ' '
      << int32_t(255.999 * pixel_color.z) << '\n';
}

constexpr float kInf = std::numeric_limits<float>::infinity();
bool ray_color(Intersect &record, Ray const &ray,
               std::vector<Sphere> const &spheres) {
  record.dist = kInf;
  Intersect intersect;
  for (auto const &sphere : spheres) {
    if (ray.intersect(intersect, sphere, 0, kInf) &&
        intersect.dist < record.dist) {
      record = intersect;
    }
  }
  if (record.dist != kInf) {
    record.color = div(add(record.normal, 1), 2);
    return true;
  }

  auto t = (ray.dir_.y + 1) / 2;
  auto white = vecf_t{1, 1, 1};
  auto blue = vecf_t{0.5, 0.7, 1};
  record.color = add(mul(white, (1.0 - t)), mul(blue, t));
  return false;
}

int32_t main() {

  auto spheres = std::vector<Sphere>();
  spheres.push_back(Sphere{vecf_t{0, 0, -1}, 0.5});
  spheres.push_back(Sphere{vecf_t{0, -100.5, -1}, 100});

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
  auto const lower_left_corner =
      sub(v1 = origin, add(div(add(v0 = horizontal, vertical), 2),
                           vecf_t{0, 0, focal_length}));
  float max_depth = 1;

  // Render

  std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

  for (int32_t j = image_height - 1; j > -1; --j) {
    for (int32_t i = 0; i < image_width; ++i) {
      auto u = float(i) / float(image_width - 1);
      auto v = float(j) / float(image_height - 1);
      // ray from origin to pixel

      auto r = Ray{
          origin,
          normalize(sub(add(add(mul(v0 = horizontal, u), mul(v1 = vertical, v)),
                            lower_left_corner),
                        origin))};
      int32_t depth = 0;
      Intersect intersect;
      // gradient
      while (ray_color(intersect, r, spheres) && depth < max_depth) {
        depth += 1;
      }
      write_color(std::cout, intersect.color);
    }
  }
}