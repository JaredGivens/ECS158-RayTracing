#pragma once
#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vec3.h"


class CudaRender
{
public:
	CudaRender() = default;
	static void Render(uint32_t width, uint32_t height, uint32_t* shared_image_data);
	__device__ static Color PerPixel(vec3 coord);

public:
	//uint32_t* shared_image_data = nullptr;
};