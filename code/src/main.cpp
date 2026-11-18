#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "culling.cuh"

// shaders
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 offset;
layout (location = 2) in int visible;

uniform mat4 vpMatrix;
uniform float time;

out float height;
out float distToCam; // Track distance from camera

void main() {
    if (visible == 0) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0); 
    } else {
        vec3 pos = aPos + offset;
        if (aPos.y > 0.0) {
            pos.x += sin(time * 2.0 + offset.x * 0.5) * 0.3;
            pos.z += cos(time * 1.5 + offset.z * 0.5) * 0.3;
        }
        gl_Position = vpMatrix * vec4(pos, 1.0);
        height = aPos.y;
        
        // Grab the perspective depth for the fog calculation
        distToCam = gl_Position.w; 
    }
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
in float height;
in float distToCam; // Receive distance
out vec4 FragColor;

void main() {
    vec3 rootColor = vec3(0.05, 0.25, 0.05);
    vec3 tipColor = vec3(0.4, 0.9, 0.3);
    vec3 grassColor = mix(rootColor, tipColor, height);
    
    // fog math
    float fogFactor = clamp((distToCam - 200.0) / 300.0, 0.0, 1.0);
    vec3 skyColor = vec3(0.5, 0.75, 0.95); // Must match glClearColor
    
    FragColor = vec4(mix(grassColor, skyColor, fogFactor), 1.0);
}
)";

// floor shaders
const char* floorVertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 vpMatrix;

out float distToCam;

void main() {
    gl_Position = vpMatrix * vec4(aPos, 1.0);
    distToCam = gl_Position.w;
}
)";

const char* floorFragmentShaderSource = R"(
#version 330 core
in float distToCam;
out vec4 FragColor;

void main() {
    vec3 dirtColor = vec3(0.25, 0.18, 0.11);
    vec3 skyColor = vec3(0.5, 0.75, 0.95);
    
    float fogFactor = clamp((distToCam - 200.0) / 300.0, 0.0, 1.0);
    FragColor = vec4(mix(dirtColor, skyColor, fogFactor), 1.0);
}
)";

// camera helper
void extractFrustumPlanes(const glm::mat4& vp, Frustum& frustum) {
    frustum.planes[0].normal.x = vp[0][3] + vp[0][0]; frustum.planes[0].normal.y = vp[1][3] + vp[1][0]; frustum.planes[0].normal.z = vp[2][3] + vp[2][0]; frustum.planes[0].distance = vp[3][3] + vp[3][0];
    frustum.planes[1].normal.x = vp[0][3] - vp[0][0]; frustum.planes[1].normal.y = vp[1][3] - vp[1][0]; frustum.planes[1].normal.z = vp[2][3] - vp[2][0]; frustum.planes[1].distance = vp[3][3] - vp[3][0];
    frustum.planes[2].normal.x = vp[0][3] + vp[0][1]; frustum.planes[2].normal.y = vp[1][3] + vp[1][1]; frustum.planes[2].normal.z = vp[2][3] + vp[2][1]; frustum.planes[2].distance = vp[3][3] + vp[3][1];
    frustum.planes[3].normal.x = vp[0][3] - vp[0][1]; frustum.planes[3].normal.y = vp[1][3] - vp[1][1]; frustum.planes[3].normal.z = vp[2][3] - vp[2][1]; frustum.planes[3].distance = vp[3][3] - vp[3][1];
    frustum.planes[4].normal.x = vp[0][3] + vp[0][2]; frustum.planes[4].normal.y = vp[1][3] + vp[1][2]; frustum.planes[4].normal.z = vp[2][3] + vp[2][2]; frustum.planes[4].distance = vp[3][3] + vp[3][2];
    frustum.planes[5].normal.x = vp[0][3] - vp[0][2]; frustum.planes[5].normal.y = vp[1][3] - vp[1][2]; frustum.planes[5].normal.z = vp[2][3] - vp[2][2]; frustum.planes[5].distance = vp[3][3] - vp[3][2];
    for (int i = 0; i < 6; ++i) {
        float length = glm::length(frustum.planes[i].normal);
        frustum.planes[i].normal /= length; frustum.planes[i].distance /= length;
    }
}

// data generator
void generateGrassData(int numObjects, float fieldSize, std::vector<AABB>& outGrass, std::vector<glm::vec3>& outOffsets, std::vector<int>& outVisibility) {
    std::random_device rd; std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-fieldSize / 2.0f, fieldSize / 2.0f);

    for (int i = 0; i < numObjects; ++i) {
        float x = dist(gen); float z = dist(gen);
        outOffsets.push_back(glm::vec3(x, 0.0f, z));

        AABB box;
        box.minExtents = glm::vec3(x - 0.25f, 0.0f, z - 0.25f);
        box.maxExtents = glm::vec3(x + 0.25f, 1.0f, z + 0.25f);
        outGrass.push_back(box);
        outVisibility.push_back(1);
    }
}

void buildSpatialGrid(std::vector<AABB>& grass, std::vector<glm::vec3>& offsets, std::vector<Chunk>& outChunks, float fieldSize, float chunkSize) {
    // Calculate how many chunks needed across one axis
    int gridDim = std::ceil(fieldSize / chunkSize);
    int numChunks = gridDim * gridDim;

    // Temp lists to hold the indices of grass that belong to each chunk
    std::vector<std::vector<int>> chunkIndices(numChunks);
    outChunks.resize(numChunks);

    // Initialize the chunk bounding boxes to extreme reverse values 
    // so they can stretch to fit the grass inside them
    for (int i = 0; i < numChunks; i++) {
        outChunks[i].bounds.minExtents = glm::vec3(1e9f);
        outChunks[i].bounds.maxExtents = glm::vec3(-1e9f);
        outChunks[i].numGrass = 0;
    }

    // Assign each blade of grass to a chunk
    for (int i = 0; i < offsets.size(); i++) {
        // Shift the coordinates so (0,0) is the bottom left of the grid
        float shiftedX = offsets[i].x + (fieldSize / 2.0f);
        float shiftedZ = offsets[i].z + (fieldSize / 2.0f);

        // Figure out which grid cell this falls into
        int gridX = std::floor(shiftedX / chunkSize);
        int gridZ = std::floor(shiftedZ / chunkSize);

        // Clamp to prevent out of bounds errors from floating point math
        gridX = std::max(0, std::min(gridX, gridDim - 1));
        gridZ = std::max(0, std::min(gridZ, gridDim - 1));

        int chunkId = gridX + (gridZ * gridDim);
        chunkIndices[chunkId].push_back(i);

        // Stretch the Chunk's bounding box to wrap around this blade of grass
        AABB& cb = outChunks[chunkId].bounds;
        AABB& gb = grass[i];
        cb.minExtents = glm::min(cb.minExtents, gb.minExtents);
        cb.maxExtents = glm::max(cb.maxExtents, gb.maxExtents);
    }

    // Rebuild the flat arrays so the memory is contiguous per chunk
    std::vector<AABB> sortedGrass;
    std::vector<glm::vec3> sortedOffsets;
    sortedGrass.reserve(grass.size());
    sortedOffsets.reserve(offsets.size());

    int currentIndex = 0;
    for (int i = 0; i < numChunks; i++) {
        outChunks[i].startIndex = currentIndex;
        outChunks[i].numGrass = chunkIndices[i].size();

        for (int originalIdx : chunkIndices[i]) {
            sortedGrass.push_back(grass[originalIdx]);
            sortedOffsets.push_back(offsets[originalIdx]);
            currentIndex++;
        }
    }

    // Replace the old random arrays with the newly sorted ones
    grass = std::move(sortedGrass);
    offsets = std::move(sortedOffsets);

    //std::cout << "Successfully grouped grass into " << numChunks << " spatial chunks." << std::endl;
}

// CPU Baseline
void run_cpu_culling(const std::vector<AABB>& boxes, std::vector<int>& visibility, const Frustum& frustum) {
    for (size_t i = 0; i < boxes.size(); ++i) {
        bool visible = true;
        for (int p = 0; p < 6; ++p) {
            glm::vec3 pt = boxes[i].minExtents;
            if (frustum.planes[p].normal.x >= 0) pt.x = boxes[i].maxExtents.x;
            if (frustum.planes[p].normal.y >= 0) pt.y = boxes[i].maxExtents.y;
            if (frustum.planes[p].normal.z >= 0) pt.z = boxes[i].maxExtents.z;

            if (glm::dot(frustum.planes[p].normal, pt) + frustum.planes[p].distance < 0) {
                visible = false;
                break;
            }
        }
        visibility[i] = visible ? 1 : 0;
    }
}

int main() {
    const int NUM_OBJECTS = 1000000;
    std::vector<AABB> grassBounds;
    std::vector<glm::vec3> grassOffsets;
    std::vector<int> gpuVisibility;

    std::cout << "Generating " << NUM_OBJECTS << " grass instances..." << std::endl;
    generateGrassData(NUM_OBJECTS, 1000.0f, grassBounds, grassOffsets, gpuVisibility);

    std::vector<Chunk> spatialGrid;
    // Set grid to 1000.0f and chunkSize to 25.0f so it perfectly matches the generation 
    // and creates exactly 1600 chunks for the GPU.
    buildSpatialGrid(grassBounds, grassOffsets, spatialGrid, 1000.0f, 25.0f);

    // Benchmarks
    std::cout << "\nRunning Scaling Benchmark" << std::endl;
    std::cout << "Objects\t\tCPU (ms)\tNaive GPU (ms)\tGrid GPU (ms)" << std::endl;
    std::cout << "-------------------------------------------------------------" << std::endl;

    glm::mat4 testProj = glm::perspective(glm::radians(60.0f), 1280.0f / 720.0f, 0.1f, 1000.0f);
    glm::mat4 testView = glm::lookAt(glm::vec3(0, 20, 150), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
    Frustum testFrustum;
    extractFrustumPlanes(testProj * testView, testFrustum);

    // Array of object counts to test scalability
    std::vector<int> testSizes = { 10000, 50000, 100000, 500000, 1000000 };

    for (int testNum : testSizes) {
        std::vector<AABB> testGrass;
        std::vector<glm::vec3> testOffsets;
        std::vector<int> testVis;
        std::vector<Chunk> testGrid;

        // Generate data for this specific scale
        generateGrassData(testNum, 1000.0f, testGrass, testOffsets, testVis);
        buildSpatialGrid(testGrass, testOffsets, testGrid, 1000.0f, 25.0f);

        std::vector<int> cpuVisibility(testNum, 1);
        std::vector<int> gpuVisibility(testNum, 1);

        // Time CPU Baseline
        auto startCPU = std::chrono::high_resolution_clock::now();
        run_cpu_culling(testGrass, cpuVisibility, testFrustum);
        auto endCPU = std::chrono::high_resolution_clock::now();
        double cpuTime = std::chrono::duration<double, std::milli>(endCPU - startCPU).count();

        // Initialize GPU memory for this specific size
        init_cuda_memory(testGrass.data(), testGrid.data(), testNum, testGrid.size());

        // Time Naive GPU
        run_naive_cuda_culling(gpuVisibility.data(), testNum, testFrustum); // Warmup
        auto startNaive = std::chrono::high_resolution_clock::now();
        run_naive_cuda_culling(gpuVisibility.data(), testNum, testFrustum);
        auto endNaive = std::chrono::high_resolution_clock::now();
        double naiveTime = std::chrono::duration<double, std::milli>(endNaive - startNaive).count();

        // Time Grid Optimized GPU
        run_cuda_culling(gpuVisibility.data(), testNum, testGrid.size(), testFrustum); // Warmup
        auto startGrid = std::chrono::high_resolution_clock::now();
        run_cuda_culling(gpuVisibility.data(), testNum, testGrid.size(), testFrustum);
        auto endGrid = std::chrono::high_resolution_clock::now();
        double gridTime = std::chrono::duration<double, std::milli>(endGrid - startGrid).count();

        printf("%-10d\t%-10.3f\t%-14.3f\t%-10.3f\n", testNum, cpuTime, naiveTime, gridTime);
    }
    std::cout << "-------------------------------------------------------------\n" << std::endl;

    // Reinitialize memory with the full 1,000,000 objects so the visualizer works properly
    init_cuda_memory(grassBounds.data(), spatialGrid.data(), NUM_OBJECTS, spatialGrid.size());

    // GLFW & OPENGL SETUP
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Frustum Culling Visualizer", NULL, NULL);
    glfwMakeContextCurrent(window);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    glEnable(GL_DEPTH_TEST);

    // compile shaders
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glUseProgram(shaderProgram);

    // setup vao vbo
    // builds floor
    GLuint floorVS = glCreateShader(GL_VERTEX_SHADER); glShaderSource(floorVS, 1, &floorVertexShaderSource, NULL); glCompileShader(floorVS);
    GLuint floorFS = glCreateShader(GL_FRAGMENT_SHADER); glShaderSource(floorFS, 1, &floorFragmentShaderSource, NULL); glCompileShader(floorFS);
    GLuint floorShaderProgram = glCreateProgram(); glAttachShader(floorShaderProgram, floorVS); glAttachShader(floorShaderProgram, floorFS); glLinkProgram(floorShaderProgram);

    float floorVertices[] = {
        -500.0f, 0.0f, -500.0f,   500.0f, 0.0f, -500.0f,   -500.0f, 0.0f,  500.0f,
         500.0f, 0.0f,  500.0f,  -500.0f, 0.0f,  500.0f,    500.0f, 0.0f, -500.0f
    };
    GLuint floorVAO, floorVBO;
    glGenVertexArrays(1, &floorVAO); glGenBuffers(1, &floorVBO);
    glBindVertexArray(floorVAO); glBindBuffer(GL_ARRAY_BUFFER, floorVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(floorVertices), floorVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    float grassGeometry[] = { 0.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f }; // A simple vert line

    GLuint VAO, geometryVBO, offsetVBO, visibilityVBO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    // Base geometry
    glGenBuffers(1, &geometryVBO);
    glBindBuffer(GL_ARRAY_BUFFER, geometryVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(grassGeometry), grassGeometry, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    // Instance Offsets (Positions)
    glGenBuffers(1, &offsetVBO);
    glBindBuffer(GL_ARRAY_BUFFER, offsetVBO);
    glBufferData(GL_ARRAY_BUFFER, grassOffsets.size() * sizeof(glm::vec3), grassOffsets.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glVertexAttribDivisor(1, 1); // Tell OpenGL this is per instance data

    // Visibility Flags (Dynamic)
    glGenBuffers(1, &visibilityVBO);
    glBindBuffer(GL_ARRAY_BUFFER, visibilityVBO);
    glBufferData(GL_ARRAY_BUFFER, gpuVisibility.size() * sizeof(int), gpuVisibility.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(2);
    glVertexAttribIPointer(2, 1, GL_INT, sizeof(int), (void*)0);
    glVertexAttribDivisor(2, 1);

    // render loop
    glm::mat4 projection = glm::perspective(glm::radians(60.0f), 1280.0f / 720.0f, 0.1f, 1000.0f);
    int vpLoc = glGetUniformLocation(shaderProgram, "vpMatrix");

    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.5f, 0.75f, 0.95f, 1.0f); // Sky Blue Background
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Animate the camera in a circle
        float time = glfwGetTime() * 0.2f;
        glm::vec3 camPos(sin(time) * 150.0f, 20.0f, cos(time) * 150.0f);
        glm::mat4 view = glm::lookAt(camPos, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
        glm::mat4 vpMatrix = projection * view;

        glUniformMatrix4fv(vpLoc, 1, GL_FALSE, glm::value_ptr(vpMatrix));

        // Draw the dirt floor
        glUseProgram(floorShaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(floorShaderProgram, "vpMatrix"), 1, GL_FALSE, glm::value_ptr(vpMatrix));
        glBindVertexArray(floorVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // Switch back to the grass shader
        glUseProgram(shaderProgram);

        int timeLoc = glGetUniformLocation(shaderProgram, "time");
        glUniform1f(timeLoc, time);

        // Update Camera Frustum Math
        Frustum camFrustum;
        extractFrustumPlanes(vpMatrix, camFrustum);

        // RUN CUDA KERNEL
        // RUN THE OPTIMIZED CHUNK KERNEL
        run_cuda_culling(gpuVisibility.data(), NUM_OBJECTS, spatialGrid.size(), camFrustum);

        // Update the OpenGL visibility buffer with the new CUDA results
        glBindBuffer(GL_ARRAY_BUFFER, visibilityVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, gpuVisibility.size() * sizeof(int), gpuVisibility.data());

        // Draw 1,000,000 blades of grass
        glBindVertexArray(VAO);
        glDrawArraysInstanced(GL_LINES, 0, 2, NUM_OBJECTS);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}