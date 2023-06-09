#pragma once

#include <glm/glm.hpp>

#include <vector>

struct Material
{
	glm::vec3 Albedo{ 1.0f };
	float Roughness = 1.0f;
	float Metallic = 0.0f;
};

struct Sphere
{
	glm::vec3 Position{0.0f};
	float Radius = 0.5f;

	int MaterialIndex = 0;
};

struct Scene
{
	glm::vec3 skycolor = glm::vec3(0.2f, 0.2f, 0.2f);
	std::vector<Sphere> Spheres;
    std::vector<Material> Materials;
};