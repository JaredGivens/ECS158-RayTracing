#include "CudaRender.h"
#include "vec3.h"
#include <stdio.h>
#include <math.h>

vec4* dev_accumulation = nullptr;
u32 dev_acc_pixel_count = 0;
u32* dev_framebuffer = nullptr; // the frame buffer that device writes to



const __device__ float CudaRenderInfo::Float() const
{
        float rand =  (float)curand_uniform(curandState);
        return rand;
}

const __device__ glm::vec3 CudaRenderInfo::randVec3() const
{
    return glm::vec3(Float(), Float(), Float());
}

const __device__ glm::vec3 CudaRenderInfo::randVec3(float min, float max) const
{
    return glm::vec3(Float() * (max - min) + min,
     Float() * (max - min) + min,
     Float() * (max - min) + min);
}

__global__ void raytraceKernel(u32* device_image_data, vec4* dev_accumulation, 
        Sphere* spheres, u32 sphere_count,
        Material* materials, u32 materials_count,
        u32 width, u32 height,
        vec3 cameraPosition, glm::mat4 inverseProjection,
        glm::mat4 inverseView, vec3 skycolor, u32 frameindex) {

    u32 x = threadIdx.x + blockIdx.x * blockDim.x;
    u32 y = threadIdx.y + blockIdx.y * blockDim.y;
    u32 fb_index = x + y * width; //index in framebuffer/accumulation buffer
    //rand
    curandState state;
    curand_init(clock64() * x * height + y, 912839, 0, &state);
    
    if ((x >= width) || (y>= height)) return;
    CudaRenderInfo cudaRenderInfo;
    cudaRenderInfo.materials = materials;
    cudaRenderInfo.materials_count = materials_count;
    cudaRenderInfo.spheres = spheres;
    cudaRenderInfo.sphere_count = sphere_count;
    cudaRenderInfo.width = width;
    cudaRenderInfo.height = height;
    cudaRenderInfo.cameraPosition = cameraPosition;
    cudaRenderInfo.inverseProjection = inverseProjection;
    cudaRenderInfo.inverseView = inverseView;
    cudaRenderInfo.curandState = &state;
    cudaRenderInfo.skycolor = skycolor;

    Color color = CudaRender::PerPixel(x, y, cudaRenderInfo);

    //reset accumlation if frameindex == 1;
    if(frameindex == 1)
    {
        dev_accumulation[fb_index] = vec4{0.f}; // hopefuly sets to zero matrix?
    }
    dev_accumulation[fb_index] += color.to_vec4();

    glm::vec4 accumulatedColor = dev_accumulation[fb_index];
    accumulatedColor /= (float)frameindex;

    accumulatedColor = glm::clamp(accumulatedColor, glm::vec4(0.0f), glm::vec4(1.0f));
    Color accColor = Color{accumulatedColor};
    device_image_data[fb_index] = accColor.ConvertToRGBA();
    //if(coord[1] > 0.5f) device_image_data[x + y * width] = 0xffffffff;
}

__device__ Color CudaRender::PerPixel(uint32_t x, uint32_t y, const CudaRenderInfo& renderInfo)
{
    //camera code 
    glm::vec2 coord = { (float)x / (float)renderInfo.width, (float)y / (float)renderInfo.height };
    coord = coord * 2.0f - 1.0f; // -1 -> 1
    glm::vec4 target = renderInfo.inverseProjection * glm::vec4(coord.x, coord.y, 1, 1);
    glm::vec3 rayDirection = glm::vec3(renderInfo.inverseView * glm::vec4(
        glm::normalize(glm::vec3(target) / target.w), 0)); // World space

    Ray ray; 
    ray.Origin = renderInfo.cameraPosition;
    ray.Direction = rayDirection;

	glm::vec3 color(0.0f);
	float multiplier = 1.0f;

	int bounces = 8;
	for (int i = 0; i < bounces; i++)
	{
		CudaRender::HitPayload payload = CudaRender::TraceRay(ray, renderInfo);
		if (payload.HitDistance < 0.0f)
		{
			glm::vec3 skyColor = renderInfo.skycolor;
			color += skyColor * multiplier;
			break;
		}

		const Sphere& sphere = renderInfo.spheres[payload.ObjectIndex];
		const Material& material = renderInfo.materials[sphere.MaterialIndex];

		glm::vec3 lightDir = glm::normalize(glm::vec3(-1, -1, -1));
        glm::vec3 halfway = -glm::normalize(lightDir + ray.Direction);
		float lightIntensity = glm::max(glm::dot(payload.WorldNormal,
            lightDir), 0.0f); // == cos(angle)
        float specIntensity = glm::pow(lightIntensity, 4.0f);
        lightIntensity += specIntensity * (1.0f - material.Roughness);

		glm::vec3 sphereColor = material.Albedo;
		sphereColor *= lightIntensity;
		color += sphereColor * multiplier;

		multiplier *= 0.5f * material.Metallic;

		ray.Origin = payload.WorldPosition + payload.WorldNormal * 0.0001f;
        glm::vec3 scatNorm = glm::normalize(payload.WorldNormal + material.Roughness *
            renderInfo.randVec3(-0.5f, 0.5f));
		ray.Direction = glm::reflect(ray.Direction, scatNorm);
	}

    return Color(color.r, color.g, color.b, 1.f, 0);

}
__device__ CudaRender::HitPayload CudaRender::TraceRay(const Ray& ray,
    const CudaRenderInfo& renderInfo)
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

        // Second hit distance (currently unused)
		// float t0 = (-b + glm::sqrt(discriminant)) / (2.0f * a); 
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

CudaRender::HitPayload CudaRender::ClosestHit(const Ray& ray, float hitDistance,
    int objectIndex, const CudaRenderInfo& renderInfo)
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
cudaError CudaRender::Render(u32 width, u32 height, u32* host_image_data,
    vec4* host_accumulation, const Scene& scene, const Camera& camera) {

    u32 pixel_count = width * height;
    Sphere* dev_spheres = nullptr;
    Material* dev_materials = nullptr;
    u32 sphere_count = scene.Spheres.size();
    u32 materials_count = scene.Materials.size();
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    
    //handle window resizing.
    if(pixel_count != dev_acc_pixel_count){
        cudaFree(dev_framebuffer);
        cudaFree(dev_accumulation);
        dev_acc_pixel_count = 0;
        dev_accumulation = nullptr;
    }

    // Allocate GPU buffers  (3 input, 1 output).
    if(nullptr == dev_accumulation){
        dev_acc_pixel_count = pixel_count;
        cudaStatus = cudaMalloc((void**)&dev_accumulation, pixel_count * sizeof(vec4));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc (dev_accumulation) failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_framebuffer, pixel_count * sizeof(u32));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc (dev_framebuffer) failed!");
            goto Error;
        }

    }

    cudaStatus = cudaMalloc((void**)&dev_spheres, sphere_count * sizeof(Sphere));
    if (cudaStatus != cudaSuccess) {
       fprintf(stderr, "cudaMalloc (dev_spheres) failed!");
       goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_materials, materials_count * sizeof(Sphere));
    if (cudaStatus != cudaSuccess) {
       fprintf(stderr, "cudaMalloc (dev_spheres) failed!");
       goto Error;
    }

    // Copy spheres array from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_spheres, &scene.Spheres.front(),
        sphere_count * sizeof(Sphere), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
       fprintf(stderr, "cudaMemcpy (scene.Spheres) failed!");
       goto Error;
    }

    //copy materials array to gpu buffers
    cudaStatus = cudaMemcpy(dev_materials, &scene.Materials.front(), 
        materials_count * sizeof(Material), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
       fprintf(stderr, "cudaMemcpy failed!");
       goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    //int blockSize = 1024;
    //int numBlocks = (pixel_count + blockSize - 1) / blockSize;
    int tx = 8;
    int ty = 8;
    // Render our buffer
    dim3 blocks(width / tx, height/ ty);
    dim3 threads(tx, ty);
    auto inverseProjection = camera.GetInverseProjection();
    auto inverseView = camera.GetInverseView();
    auto cameraPos = camera.GetPosition();

    raytraceKernel <<<blocks, threads>>> (dev_framebuffer, dev_accumulation,
     dev_spheres, sphere_count, dev_materials, materials_count, width, height, 
     cameraPos, inverseProjection, inverseView, scene.skycolor, scene.frameindex);

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

    // Copy outputs from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(host_image_data, dev_framebuffer, pixel_count * sizeof(u32), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_spheres);
    cudaFree(dev_materials);

    return cudaStatus;
}
