#include "CudaRender.h"
#include "vec3.h"
#include <stdio.h>


__global__ void renderKernel(u32* device_image_data, Sphere* spheres, u32 sphere_count, u32 width, u32 height, vec3 cameraPosition, glm::mat4 inverseProjection, glm::mat4 inverseView) {

    //printf("From cuda code\n");
    u32 x = threadIdx.x + blockIdx.x * blockDim.x;
    u32 y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= width) || (y>= height)) return;
    CudaRenderInfo cudaRenderInfo;
    cudaRenderInfo.spheres = spheres;
    cudaRenderInfo.sphere_count = sphere_count;
    cudaRenderInfo.width = width;
    cudaRenderInfo.height = height;
    cudaRenderInfo.cameraPosition = cameraPosition;
    cudaRenderInfo.inverseProjection = inverseProjection;
    cudaRenderInfo.inverseView = inverseView;
    //debug code
    //if (x != 0 || y != 0) return;
    //auto sphereColor = spheres[0].Albedo;
    //printf("Albedo1: r%f g%f b%f a%f\n", sphereColor.r, sphereColor.g, sphereColor.b, 1.0f);
    //sphereColor = spheres[1].Albedo;
    //printf("Albedo0: r%f g%f b%f a%f\n", sphereColor.r, sphereColor.g, sphereColor.b, 1.0f);

    Color color = CudaRender::PerPixel(x, y, cudaRenderInfo);
    color = color.Clamp(0, 1);
    device_image_data[x + y * width] = color.ConvertToRGBA();
    //if(coord[1] > 0.5f) device_image_data[x + y * width] = 0xffffffff;
}

cudaError_t addWithCuda(u32* out, u32 width, u32 height, const Scene& scene, const Camera& camera);

void CudaRender::Render(uint32_t width, uint32_t height, uint32_t* host_image_data, const Scene& scene, const Camera& camera) {
	addWithCuda(host_image_data, width, height, scene, camera);
}

__device__ Color CudaRender::PerPixel(uint32_t x, uint32_t y, const CudaRenderInfo& renderInfo)
{
    //camera code 
    glm::vec2 coord = { (float)x / (float)renderInfo.width, (float)y / (float)renderInfo.height };
    coord = coord * 2.0f - 1.0f; // -1 -> 1
    glm::vec4 target = renderInfo.inverseProjection * glm::vec4(coord.x, coord.y, 1, 1);
    glm::vec3 rayDirection = glm::vec3(renderInfo.inverseView * glm::vec4(glm::normalize(glm::vec3(target) / target.w), 0)); // World space

    Ray ray; 
    ray.Origin = renderInfo.cameraPosition;
    ray.Direction = rayDirection;

	glm::vec3 color(0.0f);
	float multiplier = 1.0f;

	int bounces = 18;
	for (int i = 0; i < bounces; i++)
	{
		CudaRender::HitPayload payload = CudaRender::TraceRay(ray, renderInfo);
		if (payload.HitDistance < 0.0f)
		{
			glm::vec3 skyColor = glm::vec3(0.0f, 0.0f, 0.0f);
			color += skyColor * multiplier;
			break;
		}

		glm::vec3 lightDir = glm::normalize(glm::vec3(-1, -1, -1));
		float lightIntensity = glm::max(glm::dot(payload.WorldNormal, -lightDir), 0.0f); // == cos(angle)

		const Sphere& sphere = renderInfo.spheres[payload.ObjectIndex];
		glm::vec3 sphereColor = sphere.Albedo;
		sphereColor *= lightIntensity;
		color += sphereColor * multiplier;

		multiplier *= 0.7f;

		ray.Origin = payload.WorldPosition + payload.WorldNormal * 0.0001f;
		ray.Direction = glm::reflect(ray.Direction, payload.WorldNormal);
	}

    return Color(color.r, color.g, color.b, 1.f, 0);

}
__device__ CudaRender::HitPayload CudaRender::TraceRay(const Ray& ray, const CudaRenderInfo& renderInfo)
{
    const Sphere* spheres = renderInfo.spheres;
    u32 sphere_count = renderInfo.sphere_count;
	// (bx^2 + by^2)t^2 + (2(axbx + ayby))t + (ax^2 + ay^2 - r^2) = 0
	// where
	// a = ray origin
	// b = ray direction
	// r = radius
	// t = hit distance

    int closestSphere = -1;
	float hitDistance = FLT_MAX;

	for (u32 i=0; i<sphere_count; i++)
	{
        const Sphere& sphere = spheres[i];
		glm::vec3 origin = ray.Origin - sphere.Position;

		float a = glm::dot(ray.Direction, ray.Direction);
		float b = 2.0f * glm::dot(origin, ray.Direction);
		float c = glm::dot(origin, origin) - sphere.Radius * sphere.Radius;

		// Quadratic forumula discriminant:
		// b^2 - 4ac

		float discriminant = b * b - 4.0f * a * c;
		if (discriminant < 0.0f)
			continue;

		// Quadratic formula:
		// (-b +- sqrt(discriminant)) / 2a

		// float t0 = (-b + glm::sqrt(discriminant)) / (2.0f * a); // Second hit distance (currently unused)
		float closestT = (-b - glm::sqrt(discriminant)) / (2.0f * a);
        if (closestT > 0.0f && closestT < hitDistance)
        {
            hitDistance = closestT;
            closestSphere = (int)i;
        }
    }

    if (closestSphere < 0)
        return Miss(ray, renderInfo);

    return ClosestHit(ray, hitDistance, closestSphere, renderInfo);
}

CudaRender::HitPayload CudaRender::ClosestHit(const Ray& ray, float hitDistance, int objectIndex, const CudaRenderInfo& renderInfo)
{
	CudaRender::HitPayload payload;
	payload.HitDistance = hitDistance;
	payload.ObjectIndex = objectIndex;

	const Sphere& closestSphere = renderInfo.spheres[objectIndex];

	glm::vec3 origin = ray.Origin - closestSphere.Position;
	payload.WorldPosition = origin + ray.Direction * hitDistance;
	payload.WorldNormal = glm::normalize(payload.WorldPosition);

	payload.WorldPosition += closestSphere.Position;

	return payload;
}

CudaRender::HitPayload CudaRender::Miss(const Ray& ray, const CudaRenderInfo& renderInfo)
{
	CudaRender::HitPayload payload;
	payload.HitDistance = -1.0f;
	return payload;
}

//writes to out array using cuda
cudaError_t addWithCuda(u32* out, u32 width, u32 height, const Scene& scene, const Camera& camera)
{
    u32 problem_size = width * height;
    uint32_t* dev_c = 0; // the frame buffer that device writes to
    Sphere* dev_spheres = 0;
    u32 sphere_count = scene.Spheres.size();
    cudaError_t cudaStatus;

    std::cout << "Sphere count: " << sphere_count << std::endl;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, problem_size * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_spheres, sphere_count * sizeof(Sphere));
    if (cudaStatus != cudaSuccess) {
       fprintf(stderr, "cudaMalloc failed!");
       goto Error;
    }

    //cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaMalloc failed!");
    //    goto Error;
    //}

    //// Copy input vectors from host memory to GPU buffers.
    //cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaMemcpy failed!");
    //    goto Error;
    //}

    // Copy spheres array from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_spheres, &scene.Spheres.front(), sphere_count * sizeof(Sphere), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
       fprintf(stderr, "cudaMemcpy failed!");
       goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    int blockSize = 1024;
    int numBlocks = (problem_size + blockSize - 1) / blockSize;
    printf("Num blocks: %d\n", numBlocks);
    int tx = 8;
    int ty = 8;
    // Render our buffer
    dim3 blocks(width / tx, height/ ty);
    dim3 threads(tx, ty);
    auto inverseProjection = camera.GetInverseProjection();
    auto inverseView = camera.GetInverseView();
    auto cameraPos = camera.GetPosition();
    renderKernel << <blocks, threads>> > (dev_c, dev_spheres, sphere_count, width, height, cameraPos, inverseProjection, inverseView);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(out, dev_c, problem_size * sizeof(u32), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);

    return cudaStatus;
}
