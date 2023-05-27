#include "geo.hh"
#include <iostream>
#include <random>
#include <stdint.h>
#include <vector>

class Camera {
public:
  Camera() {
    float aspect_ratio = 16.0 / 9.0;
    float viewport_height = 2.0;
    float viewport_width = aspect_ratio * viewport_height;
    float focal_length = 1.0;

    origin_ = vecf_t{0, 0, 0};
    hor_ = vecf_t{viewport_width, 0.0, 0.0};
    vert_ = vecf_t{0.0, viewport_height, 0.0};
    vecf_t hor = hor_;
    vecf_t origin = origin_;
    lower_left_ =
        sub(origin, add(div(add(hor, vert_), 2), vecf_t{0, 0, focal_length}));
  }

  Ray get_ray(float u, float v) const {
    vecf_t hor = hor_;
    vecf_t vert = vert_;
    return Ray{origin_,
               normalize(sub(add(add(mul(hor, u), mul(vert, v)), lower_left_),
                             origin_))};
  }

private:
  vecf_t origin_;
  vecf_t lower_left_;
  vecf_t hor_;
  vecf_t vert_;
};

constexpr float kInf = std::numeric_limits<float>::infinity();
bool ray_color(Intersect &record, Ray const &ray,
               std::vector<Sphere> const &spheres) {
  record.dist = kInf;
  Intersect intersect;
  for (auto const &sphere : spheres) {
    if (ray.intersect(intersect, sphere, 0.001, kInf) &&
        intersect.dist < record.dist) {
      record = intersect;
    }
  }
  if (record.dist != kInf) {
    // record.color = div(add(record.normal, 1), 2);
    record.color = vecf_t{0, 0, 0};
    return true;
  }

  auto t = (ray.dir_.y + 1) / 2;
  auto white = vecf_t{1, 1, 1};
  auto blue = vecf_t{0.5, 0.7, 1};
  record.color = add(mul(white, (1.0 - t)), mul(blue, t));
  return false;
}

vecf_t random_unit_vec(std::default_random_engine rand_eng) {
  auto static dist_unit =
      std::uniform_real_distribution<float>(-1 / sqrtf(2), 1 / sqrt(2));
  // auto static dist_unit =
  //     std::uniform_real_distribution<float>(-1, 1);
  auto res = vecf_t{dist_unit(rand_eng), dist_unit(rand_eng), dist_unit(rand_eng)};
  // technically this normalize can be removed but this is the true
  return normalize(res);
}

auto uni_dist01 = std::uniform_real_distribution<float>(0, 1);
int32_t main() {

  auto spheres = std::vector<Sphere>();
  spheres.push_back(Sphere{vecf_t{0, 0, -1}, 0.5});
  spheres.push_back(Sphere{vecf_t{0, -100.5, -1}, 100});

  // Image

  float const aspect_ratio = 16.0 / 9.0;
  int32_t const image_height = 255;
  int32_t const image_width = image_height * aspect_ratio;

  float viewport_height = 2.0;
  float viewport_width = aspect_ratio * viewport_height;
  float focal_length = 1.0;

  vecf_t v0;
  vecf_t v1;

  auto cam = Camera();
  float max_depth = 20;
  float samples = 10;
  auto rand_eng = std::default_random_engine();

  // Render

  std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

  for (int32_t j = image_height - 1; j > -1; --j) {
    for (int32_t i = 0; i < image_width; ++i) {
      auto color = vecf_t{0, 0, 0};
      for (int32_t k = 0; k < samples; ++k) {
        auto u = float(i + uni_dist01(rand_eng)) / float(image_width - 1);
        auto v = float(j + uni_dist01(rand_eng)) / float(image_height - 1);
        // ray from origin to pixel

        Ray r = cam.get_ray(u, v);
        int32_t depth = 0;
        Intersect intersect;
        // gradient
        while (ray_color(intersect, r, spheres) && depth < max_depth) {
          r.origin_ = intersect.pos;
          r.dir_ = normalize(add(intersect.normal, random_unit_vec(rand_eng)));
          depth += 1;
        }

        add(color, div(intersect.color, 1 << depth));
      }
      div(color, samples);
      color.x = sqrtf(std::clamp(color.x, 0.0f, 0.999f));
      color.y = sqrtf(std::clamp(color.y, 0.0f, 0.999f));
      color.z = sqrtf(std::clamp(color.z, 0.0f, 0.999f));
      std::cout << int32_t(255.999 * color.x) << ' '
                << int32_t(255.999 * color.y) << ' '
                << int32_t(255.999 * color.z) << std::endl;
    }
  }
}