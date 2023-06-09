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
	//Ray ray;
	//ray.Origin = camera.GetPosition();

	auto image_buf_size = sizeof(uint32_t) * m_FinalImage->GetWidth() * m_FinalImage->GetHeight();
	scene.frameindex = m_FrameIndex;
	CudaRender::Render(m_FinalImage->GetWidth(), m_FinalImage->GetHeight(), m_ImageData, m_AccumulationData, scene, camera);
	//for (uint32_t y = 0; y < m_FinalImage->GetHeight(); y++)
	//{
	//	for (uint32_t x = 0; x < m_FinalImage->GetWidth(); x++)
	//	{
	//		glm::vec2 coord = { (float)x / (float)m_FinalImage->GetWidth(), (float)y / (float)m_FinalImage->GetHeight() };
	//		coord = coord * 2.0f - 1.0f; // -1 -> 1
	//		m_ImageData[x + y * m_FinalImage->GetWidth()] = 0xffff0000;
	//		if(coord.x > 0.5f) m_ImageData[x + y * m_FinalImage->GetWidth()] = 0xffffffff;
	//	}
	//}

	m_FinalImage->SetData(m_ImageData);

	if (m_Settings.Accumulate)
	{
		m_FrameIndex++;
	}
	else
		m_FrameIndex = 1;
}

