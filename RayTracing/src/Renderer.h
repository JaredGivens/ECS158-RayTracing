#pragma once

#include "Walnut/Image.h"

#include <memory>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaRender.h"
#include "Camera.h"
#include "Ray.h"
#include "Scene.h"

class Renderer
{
public:
	struct Settings
	{
		bool Accumulate = false;
	};
public:
	Renderer() = default;

	void OnResize(uint32_t width, uint32_t height);
	void Render(Scene& scene, const Camera& camera);

	std::shared_ptr<Walnut::Image> GetFinalImage() const { return m_FinalImage; }

	void ResetFrameIndex() { m_FrameIndex = 1; }
	Settings& GetSettings() { return m_Settings; }
private:
	glm::vec4 TraceRay(const Scene& scene, const Ray& ray);
private:
	std::shared_ptr<Walnut::Image> m_FinalImage;
	Settings m_Settings;

	uint32_t* m_ImageData = nullptr;glm::vec4* m_AccumulationData = nullptr;

	uint32_t m_FrameIndex = 1;
};