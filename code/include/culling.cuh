#pragma once
#include <glm/glm.hpp>

struct AABB {
    glm::vec3 minExtents;
    glm::vec3 maxExtents;
};

struct Chunk {
    AABB bounds;
    int startIndex;
    int numGrass;
};

struct Plane {
    glm::vec3 normal;
    float distance;
};

struct Frustum {
    Plane planes[6];
};

// proto
void init_cuda_memory(const AABB* h_grass, const Chunk* h_chunks, int num_objects, int num_chunks);
void run_cuda_culling(int* h_visibility, int num_objects, int num_chunks, const Frustum& frustum);
void run_naive_cuda_culling(int* h_visibility, int num_objects, const Frustum& frustum);