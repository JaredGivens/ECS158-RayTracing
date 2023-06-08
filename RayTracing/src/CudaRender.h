#pragma once
#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vec3.h"
#include "Camera.h"
#include "Ray.h"


class CudaRender
{
public:
	CudaRender() = default;
	static void Render(uint32_t width, uint32_t height, uint32_t* shared_image_data, const Camera& camera);
	__device__ static Color TraceRay(const Ray& ray);

public:
	//uint32_t* shared_image_data = nullptr;
};