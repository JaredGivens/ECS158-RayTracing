#include "Renderer.h"
#include "Walnut/Random.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using std::cout, std::endl;

void Renderer::OnResize(uint32_t width, uint32_t height)
{
	if (m_FinalImage)
	{
		// No resize necessary
		if (m_FinalImage->GetWidth() == width && m_FinalImage->GetHeight() == height)
			return;

		m_FinalImage->Resize(width, height);
	}
	else
	{
		m_FinalImage = std::make_shared<Walnut::Image>(width, height, Walnut::ImageFormat::RGBA);
	}
	delete[] m_ImageData;
	m_ImageData = new uint32_t[width * height];


	delete[] m_AccumulationData;
	m_AccumulationData = new glm::vec4[width * height];
}

void Renderer::Render(Scene& scene, const Camera& camera)
{
	auto image_buf_size = sizeof(uint32_t) * m_FinalImage->GetWidth() * m_FinalImage->GetHeight();
	scene.frameindex = m_FrameIndex;
	CudaRender::Render(m_FinalImage->GetWidth(),
	m_FinalImage->GetHeight(),
	m_ImageData,
	m_AccumulationData,
	scene,
	camera);

	m_FinalImage->SetData(m_ImageData);

	if (m_Settings.Accumulate)
	{
		m_FrameIndex++;
	}
	else
		m_FrameIndex = 1;
}

