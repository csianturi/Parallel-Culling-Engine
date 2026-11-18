#include "culling.cuh"
#include <cuda_runtime.h>

// Global device pointers to prevent allocate memory every single frame
static AABB* d_grass = nullptr;
static Chunk* d_chunks = nullptr;
static int* d_visibility = nullptr;

__device__ bool isAABBVisible(const AABB& box, const Frustum& frustum) {
    for (int i = 0; i < 6; ++i) {
        glm::vec3 p = box.minExtents;
        if (frustum.planes[i].normal.x >= 0) p.x = box.maxExtents.x;
        if (frustum.planes[i].normal.y >= 0) p.y = box.maxExtents.y;
        if (frustum.planes[i].normal.z >= 0) p.z = box.maxExtents.z;

        if (glm::dot(frustum.planes[i].normal, p) + frustum.planes[i].distance < 0) {
            return false;
        }
    }
    return true;
}

__global__ void chunk_frustum_cull_kernel(const Chunk* chunks, const AABB* boxes, int* visibility, Frustum frustum) {
    //launch exactly 1 Block per Chunk
    int chunkIdx = blockIdx.x;
    Chunk myChunk = chunks[chunkIdx];

    // Shared memory so threads can communicate within the block
    __shared__ int chunkVisible;

    // Thread 0 tests the massive chunk bounding box
    if (threadIdx.x == 0) {
        chunkVisible = isAABBVisible(myChunk.bounds, frustum) ? 1 : 0;
    }

    // Wait for Thread 0 to finish its math
    __syncthreads();

    // If the massive chunk box is offscreen, all threads cooperatively cull the grass and exit
    if (chunkVisible == 0) {
        for (int i = threadIdx.x; i < myChunk.numGrass; i += blockDim.x) {
            visibility[myChunk.startIndex + i] = 0;
        }
        return;
    }

    // If the chunk is visible the threads cooperatively check the individual grass blades
    for (int i = threadIdx.x; i < myChunk.numGrass; i += blockDim.x) {
        int globalIdx = myChunk.startIndex + i;
        visibility[globalIdx] = isAABBVisible(boxes[globalIdx], frustum) ? 1 : 0;
    }
}

// naive kernel
__global__ void naive_frustum_cull_kernel(const AABB* boxes, int* visibility, int num_objects, Frustum frustum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_objects) {
        visibility[idx] = isAABBVisible(boxes[idx], frustum) ? 1 : 0;
    }
}

// Wrapper for the Naive Kernel
void run_naive_cuda_culling(int* h_visibility, int num_objects, const Frustum& frustum) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_objects + threadsPerBlock - 1) / threadsPerBlock;

    naive_frustum_cull_kernel << <blocksPerGrid, threadsPerBlock >> > (d_grass, d_visibility, num_objects, frustum);
    cudaDeviceSynchronize();

    cudaMemcpy(h_visibility, d_visibility, num_objects * sizeof(int), cudaMemcpyDeviceToHost);
}

// run once before render loop
void init_cuda_memory(const AABB* h_grass, const Chunk* h_chunks, int num_objects, int num_chunks) {
    cudaMalloc(&d_grass, num_objects * sizeof(AABB));
    cudaMalloc(&d_chunks, num_chunks * sizeof(Chunk));
    cudaMalloc(&d_visibility, num_objects * sizeof(int));

    cudaMemcpy(d_grass, h_grass, num_objects * sizeof(AABB), cudaMemcpyHostToDevice);
    cudaMemcpy(d_chunks, h_chunks, num_chunks * sizeof(Chunk), cudaMemcpyHostToDevice);
}

//Run every frame
void run_cuda_culling(int* h_visibility, int num_objects, int num_chunks, const Frustum& frustum) {
    int threadsPerBlock = 256;
    int blocksPerGrid = num_chunks; // 1 Block eval 1 Chunk

    chunk_frustum_cull_kernel << <blocksPerGrid, threadsPerBlock >> > (d_chunks, d_grass, d_visibility, frustum);

    cudaDeviceSynchronize();

    //only copy the lightweight visibility flags back to the CPU, saving time
    cudaMemcpy(h_visibility, d_visibility, num_objects * sizeof(int), cudaMemcpyDeviceToHost);
}