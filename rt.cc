#include <stdint.h>
#include <unistd.h>

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "geo.hh"

class Camera {
 public:
  Camera(float vfov, float aspect_ratio) {
    auto theta = vfov / 180 * kPi;
    auto h = tan(theta / 2);
    float viewport_height = 2.0 * h;
    float viewport_width = aspect_ratio * viewport_height;
    focal_length_ = 1.0;

    pos_ = vecf_t{0, 0, 0};
    rot_ = quat_t{0, 0, 0, 1};
    hor_ = vecf_t{viewport_width, 0.0, 0.0};
    vert_ = vecf_t{0.0, viewport_height, 0.0};
    update_transform();
  }

  void update_transform() {
    vecf_t hor = hor_;
    vecf_t origin = pos_;
    lower_left_ =
        sub(origin, add(div(add(hor, vert_), 2), vecf_t{0, 0, focal_length_}));
  }

  Ray get_ray(float u, float v) const {
    vecf_t vert = vert_;
    vecf_t screen;
    mul(add(add(mul(screen = hor_, u), mul(vert, v)), lower_left_), rot_);
    return Ray{pos_, normalize(sub(screen, pos_))};
  }
  vecf_t pos_;
  quat_t rot_;

 private:
  float focal_length_;
  vecf_t lower_left_;
  vecf_t hor_;
  vecf_t vert_;
};

constexpr float kInf = std::numeric_limits<float>::infinity();
bool cast_ray(Intersect &record, Ray const &ray,
              std::vector<Sphere> const &spheres) {
  record.dist = kInf;
  Intersect intersect;
  for (auto const &sphere : spheres) {
    if (ray.intersect(intersect, sphere, 0.0001, kInf) &&
        intersect.dist < record.dist) {
      record = intersect;
    }
  }
  return record.dist != kInf;
}

vecf_t refract(vecf_t const &uv, vecf_t const &n, float refraction_ratio) {
  float cos_theta = -dot(uv, n);
  vecf_t v0 = n;
  vecf_t r_out_perp = uv;
  mul(add(r_out_perp, mul(v0, cos_theta)), refraction_ratio);
  v0 = n;
  vecf_t r_out_parallel = mul(v0, -sqrt(fabs(1.0 - len_sq(r_out_perp))));
  return add(r_out_perp, r_out_parallel);
}

vecf_t &metal_reflect(vecf_t &v, Intersect const &intersect) {
  return normalize(lerp(reflect(v, intersect.normal), intersect.normal,
                        1 - intersect.mat.metalness));
}

auto uni_dist01 = std::uniform_real_distribution<float>(0, 1);
class Raytracer {
  public:
  int32_t screen_height;
  int32_t screen_width;
  int32_t samples;
  Camera cam;
  std::vector<Sphere> spheres;
  int32_t ray_depth;
  std::default_random_engine rand_eng;

  void raytrace_frame() {
    for (int32_t j = screen_height - 1; j > -1; --j) {
      for (int32_t i = 0; i < screen_width; ++i) {
        auto pixel_color = vecf_t{0, 0, 0};
        for (int32_t k = 0; k < samples; ++k) {
          auto u =
              float(i + uni_dist01(rand_eng)) / float(screen_width - 1);
          auto v =
              float(j + uni_dist01(rand_eng)) / float(screen_height - 1);
          // ray from origin to pixel

          Ray ray = cam.get_ray(u, v);
          int32_t depth = 0;
          auto ray_color = vecf_t{1, 1, 1};
          Intersect intersect;
          while (cast_ray(intersect, ray, spheres) && depth < ray_depth) {
            ray.origin_ = intersect.pos;

            if (intersect.mat.refraction > 0.1) {
              // material can refract
              float refraction_ratio = intersect.front
                                           ? 1 / intersect.mat.refraction
                                           : intersect.mat.refraction;
              float cos_theta = -dot(ray.dir_, intersect.normal);
              float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

              // Use Schlick's approximation for reflectance.
              auto reflectance =
                  (1 - refraction_ratio) / (1 + refraction_ratio);
              reflectance *= reflectance;
              reflectance += (1 - reflectance) * pow((1 - cos_theta), 5);

              if (refraction_ratio * sin_theta > 1.0 ||
                  reflectance > uni_dist01(rand_eng)) {
                metal_reflect(ray.dir_, intersect);
              } else {
                // refract
                vecf_t v0 = intersect.normal;
                mul(add(ray.dir_, mul(v0, cos_theta)), refraction_ratio);
                vecf_t parallel = mul(v0 = intersect.normal,
                                      -sqrt(fabs(1.0 - len_sq(ray.dir_))));
                add(ray.dir_, parallel);
              }
            } else {
              metal_reflect(ray.dir_, intersect);
            }

            // roughness fuzzing
            auto static dist_unit = std::uniform_real_distribution<float>(
                -1 / sqrtf(2), 1 / sqrtf(2));
            auto scatter =
                vecf_t{dist_unit(rand_eng), dist_unit(rand_eng),
                       dist_unit(rand_eng)};

            // constrain to hemisphere
            if (dot(scatter, intersect.normal) > 0) {
              normalize(add(ray.dir_, mul(scatter, intersect.mat.roughness)));
            } else {
              normalize(add(ray.dir_, mul(scatter, -intersect.mat.roughness)));
            }

            mul(ray_color, mul(intersect.mat.color, intersect.mat.specular));
            depth += 1;
          }

          // skybox color
          auto white = vecf_t{1, 1, 1};
          mul(ray_color,
              lerp(white, vecf_t{0.5, 0.7, 1}, (ray.dir_.y + 1) / 2));

          add(pixel_color, ray_color);
        }
        div(pixel_color, samples);
        std::cout << 256 * sqrtf(std::clamp(pixel_color.x, 0.0f, 0.999f))
                  << ' ';
        std::cout << 256 * sqrtf(std::clamp(pixel_color.y, 0.0f, 0.999f))
                  << ' ';
        std::cout << 256 * sqrtf(std::clamp(pixel_color.z, 0.0f, 0.999f))
                  << std::endl;
      }
    }
  }
};

int32_t main() {
  Material yellow_diff = {
      vecf_t{.9, .9, .2}, 0, .5, 1, 0,
  };

  Material blue_metal = {
      vecf_t{.6, .6, .9}, 1, .5, .1, 0,
  };

  Material glass = {
      vecf_t{1, 1, 1}, 0, 1, 0, 1.5,
  };
  auto spheres = std::vector<Sphere>();
  spheres.push_back(Sphere{vecf_t{-1, 0, -1}, 0.5, yellow_diff});
  spheres.push_back(Sphere{vecf_t{0, 0, -1}, 0.5, blue_metal});
  spheres.push_back(Sphere{vecf_t{1, 0, -1}, 0.5, glass});
  spheres.push_back(Sphere{vecf_t{1, 0, -1}, -0.4, glass});
  spheres.push_back(Sphere{vecf_t{0, -100.5, -1}, 100, yellow_diff});

  float const aspect_ratio = 16.0 / 9.0;
  int32_t const screen_height = 255;
  int32_t const screen_width = screen_height * aspect_ratio;

  auto raytracer = Raytracer{
      .screen_height = screen_height,
      .screen_width = screen_width,
      .samples = 50,
      .cam = Camera(90, aspect_ratio),
      .spheres = spheres,
      .ray_depth = 20,
      .rand_eng = std::default_random_engine(),
  };

  std::cout << "P3\n" << screen_width << ' ' << screen_height << "\n255\n";
  raytracer.raytrace_frame();
}