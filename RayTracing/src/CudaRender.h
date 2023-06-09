#pragma once
#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vec3.h"
#include "Camera.h"
#include "Ray.h"
#include "Scene.h"
#include "curand_kernel.h"

#include "Walnut/Random.h"

typedef uint32_t u32;

class CudaRenderInfo {
public:
	const __device__ glm::vec3 randVec3(float min, float max) const;
	const __device__ float Float() const ;
	const __device__ glm::vec3 randVec3() const;

public:
	const Material* materials;
	u32 materials_count; 
	const Sphere* spheres;
	u32 sphere_count;
	u32 width;
	u32 height;
	vec3 cameraPosition;
	glm::mat4 inverseProjection;
	glm::mat4 inverseView;
	curandState* curandState;
	vec3 skycolor;
};

class CudaRender
{
public:
	CudaRender() = default;
	static void Render(uint32_t width, uint32_t height, uint32_t* shared_image_data, const Scene& scene, const Camera& camera);
	//__device__ static Color TraceRay(const Sphere* spheres, uint32_t sphere_count, const Ray& ray);
	__device__ static Color CudaRender::PerPixel(uint32_t x, uint32_t y, const CudaRenderInfo& renderinfo);

private: 
	struct HitPayload
	{
		float HitDistance;
		glm::vec3 WorldPosition;
		glm::vec3 WorldNormal;

		int ObjectIndex;
	};

	__device__ static HitPayload TraceRay(const Ray& ray, const CudaRenderInfo& renderinfo);
	__device__ static HitPayload ClosestHit(const Ray& ray, float hitDistance, int objectIndex, const CudaRenderInfo& renderinfo);
	__device__ static HitPayload Miss(const Ray& ray, const CudaRenderInfo& renderinfo);
	public:
};
