#include <SDL.h>
#include <stdint.h>
#include <unistd.h>

#include <iostream>
#include <random>
#include <vector>

#include "geo.hh"

class Camera {
 public:
  Camera(float aspect_ratio) {
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
struct Raytracer {
  int32_t screen_height;
  int32_t screen_width;
  SDL_Renderer *renderer;
  int32_t samples;
  Camera cam;
  std::vector<Sphere> spheres;
  int32_t ray_depth;
  std::default_random_engine rand_eng;
};

void raytrace_frame(Raytracer &rt) {
  SDL_RenderClear(rt.renderer);
  for (int32_t j = rt.screen_height - 1; j > -1; --j) {
    for (int32_t i = 0; i < rt.screen_width; ++i) {
      auto pixel_color = vecf_t{0, 0, 0};
      for (int32_t k = 0; k < rt.samples; ++k) {
        auto u =
            float(i + uni_dist01(rt.rand_eng)) / float(rt.screen_width - 1);
        auto v =
            float(j + uni_dist01(rt.rand_eng)) / float(rt.screen_height - 1);
        // ray from origin to pixel

        Ray ray = rt.cam.get_ray(u, v);
        int32_t depth = 0;
        auto ray_color = vecf_t{1, 1, 1};
        Intersect intersect;
        while (cast_ray(intersect, ray, rt.spheres) && depth < rt.ray_depth) {
          ray.origin_ = intersect.pos;

          // this is not neccesary
          if (intersect.mat.refraction > 0.1) {
            float refraction_ratio = intersect.front
                                         ? 1 / intersect.mat.refraction
                                         : intersect.mat.refraction;
            float cos_theta = -dot(ray.dir_, intersect.normal);
            // float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
            // if (refraction_ratio * sin_theta > 1.0) {
              // metal_reflect(ray.dir_, intersect);
            // } else {
            vecf_t v0 = intersect.normal;
            mul(add(ray.dir_, mul(v0, cos_theta)), refraction_ratio);
            vecf_t parallel =
                mul(v0 = intersect.normal, -sqrt(fabs(1.0 -
                len_sq(ray.dir_))));
            add(ray.dir_, parallel);
            // }
          } else {
            metal_reflect(ray.dir_, intersect);

            // roughness fuzzing
            auto static dist_unit =
                std::uniform_real_distribution<float>(-1/sqrtf(2), 1/sqrtf(2));
            auto scatter =
                vecf_t{dist_unit(rt.rand_eng), dist_unit(rt.rand_eng),
                       dist_unit(rt.rand_eng)};

            if (dot(scatter, intersect.normal) > 0) {
              normalize(add(ray.dir_, mul(scatter, intersect.mat.roughness)));
            }else {
              normalize(add(ray.dir_, mul(scatter, -intersect.mat.roughness)));
            }
          }

          mul(ray_color, mul(intersect.mat.color, intersect.mat.specular));
          depth += 1;
        }

        // skybox color
        auto white = vecf_t{1, 1, 1};
        mul(ray_color, lerp(white, vecf_t{0.5, 0.7, 1},(ray.dir_.y + 1) / 2));


        add(pixel_color, ray_color);
      }
      div(pixel_color, rt.samples);
      int32_t r = 256 * sqrtf(std::clamp(pixel_color.x, 0.0f, 0.999f));
      int32_t g = 256 * sqrtf(std::clamp(pixel_color.y, 0.0f, 0.999f));
      int32_t b = 256 * sqrtf(std::clamp(pixel_color.z, 0.0f, 0.999f));
      SDL_SetRenderDrawColor(rt.renderer, r, g, b, 0);
      SDL_RenderDrawPoint(rt.renderer, i, rt.screen_height - j);
    }
  }
  SDL_RenderPresent(rt.renderer);
}

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
  spheres.push_back(Sphere{vecf_t{0, 0, -1}, 0.5, glass});
  spheres.push_back(Sphere{vecf_t{1, 0, -1}, 0.5, blue_metal});
  spheres.push_back(Sphere{vecf_t{0, -100.5, -1}, 100, yellow_diff});

  float const aspect_ratio = 16.0 / 9.0;
  int32_t const screen_height = 255;
  int32_t const screen_width = screen_height * aspect_ratio;

  // Initialize SDL
  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    std::cerr << "SDL Init Error:\n" << SDL_GetError() << std::endl;
    exit(1);
  }
  SDL_Window *window =
      SDL_CreateWindow("Raytracing", SDL_WINDOWPOS_CENTERED,
                       SDL_WINDOWPOS_CENTERED, screen_width, screen_height, 0);
  if (!window) {
    std::cerr << "CreateWindow Error:\n" << SDL_GetError() << std::endl;
    exit(1);
  }
  SDL_Renderer *renderer =
      SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

  if (!renderer) {
    std::cerr << "CreateRenderer Error:\n" << SDL_GetError() << std::endl;
    exit(1);
  }

  auto raytracer = Raytracer{
      .screen_height = screen_height,
      .screen_width = screen_width,
      .renderer = renderer,
      .samples = 50,
      .cam = Camera(aspect_ratio),
      .spheres = spheres,
      .ray_depth = 20,
      .rand_eng = std::default_random_engine(),
  };
  raytrace_frame(raytracer);

  // Render
  SDL_Event e;
  bool quit = false;
  while (!quit) {
    while (SDL_PollEvent(&e)) {
      if (e.type == SDL_QUIT) {
        quit = true;
      }
    }
  }
  SDL_DestroyWindow(window);
  // std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
  // We have to destroy the renderer, same as with the window.
  SDL_DestroyRenderer(renderer);
  SDL_Quit();
}