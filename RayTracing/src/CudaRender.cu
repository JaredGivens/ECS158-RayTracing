#include "CudaRender.h"
#include "vec3.h"
#include <stdio.h>


typedef uint32_t u32;
__global__ void renderKernel(u32* device_image_data, u32 width, u32 height, vec3 cameraPosition, glm::mat4 inverseProjection, glm::mat4 inverseView) {

    //printf("From cuda code\n");
    u32 x = threadIdx.x + blockIdx.x * blockDim.x;
    u32 y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= width) || (y>= height)) return;

    //camera code 
    glm::vec2 coord = { (float)x / (float)width, (float)y / (float)height };
    coord = coord * 2.0f - 1.0f; // -1 -> 1
    glm::vec4 target = inverseProjection * glm::vec4(coord.x, coord.y, 1, 1);
    glm::vec3 rayDirection = glm::vec3(inverseView * glm::vec4(glm::normalize(glm::vec3(target) / target.w), 0)); // World space

    Ray ray; 
    ray.Origin = cameraPosition;
    ray.Direction = rayDirection;
    Color color = CudaRender::TraceRay(ray);

    color = color.Clamp(0, 1);
    device_image_data[x + y * width] = color.ConvertToRGBA();
    //if(coord[1] > 0.5f) device_image_data[x + y * width] = 0xffffffff;
}

cudaError_t addWithCuda(u32* out, u32 width, u32 height, const Camera& camera);

void CudaRender::Render(uint32_t width, uint32_t height, uint32_t* host_image_data, const Camera& camera) {
	addWithCuda(host_image_data, width, height, camera);
}

__device__ Color CudaRender::TraceRay(const Ray& ray)
{
	float radius = 0.5f;
	// rayDirection = glm::normalize(rayDirection);

	// (bx^2 + by^2)t^2 + (2(axbx + ayby))t + (ax^2 + ay^2 - r^2) = 0
	// where
	// a = ray origin
	// b = ray direction
	// r = radius
	// t = hit distance

    float a = glm::dot(ray.Direction, ray.Direction);
    float b = 2.0f * glm::dot(ray.Origin, ray.Direction);
    float c = glm::dot(ray.Origin, ray.Origin) - radius * radius;

	// Quadratic forumula discriminant:
	// b^2 - 4ac

	float discriminant = b * b - 4.0f * a * c;
    if (discriminant < 0.0f)
        return Color(0, 0, 0, 1);

    // Quadratic formula:
    // (-b +- sqrt(discriminant)) / 2a

    float closestT = (-b - sqrt(discriminant)) / (2.0f * a);
    float t0 = (-b + sqrt(discriminant)) / (2.0f * a); // Second hit distance (currently unused)

    glm::vec3 hitPoint = ray.Origin + ray.Direction * closestT;
    vec3 normal = glm::normalize(hitPoint);

    vec3 lightDir = glm::normalize(vec3(-1, -0, -1));
    float lightIntensity = std::fmax(0.0f, dot(normal, -lightDir)); // == cos(angle)

    vec3 sphereColor(1, 0, 1);
    sphereColor *= lightIntensity;

    return Color(sphereColor.r, sphereColor.g, sphereColor.b, 1.0f);
}

//writes to out array using cuda
cudaError_t addWithCuda(u32* out, u32 width, u32 height, const Camera& camera)
{
    u32 problem_size = width * height;
    uint32_t* dev_c = 0;
    cudaError_t cudaStatus;

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

    //cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaMalloc failed!");
    //    goto Error;
    //}

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
    renderKernel << <blocks, threads>> > (dev_c, width, height, cameraPos, inverseProjection, inverseView);

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
