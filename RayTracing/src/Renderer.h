#pragma once

#include "Walnut/Image.h"

#include <memory>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaRender.h"

class Renderer
{
public:
	Renderer() = default;

	void OnResize(uint32_t width, uint32_t height);
	void Render();

	std::shared_ptr<Walnut::Image> GetFinalImage() const { return m_FinalImage; }
private:
	std::shared_ptr<Walnut::Image> m_FinalImage;
	uint32_t* m_ImageData = nullptr;
};