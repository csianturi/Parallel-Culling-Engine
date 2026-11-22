# CUDA Frustum Culling Benchmark

A C++ and CUDA demo that compares three ways to determine which grass instances fall inside a camera's view. It prints a scaling benchmark in the terminal, then opens an OpenGL visualizer with an orbiting camera and one million grass instances.

## What it measures

Each method tests axis-aligned bounding boxes against the six planes of the camera frustum.

| Method | How it works |
| --- | --- |
| CPU | Tests each grass instance in a serial loop. |
| Naive GPU | Assigns one CUDA thread to each grass instance. |
| Grid GPU | Tests spatial chunks first. Instances in rejected chunks skip the individual frustum test. |

The benchmark runs at 10,000, 50,000, 100,000, 500,000, and 1,000,000 instances. It prints wall-clock milliseconds for one timed call per method at each size, with a warm-up call before each GPU measurement. GPU timings include synchronization and copying the visibility flags back to the CPU. Scene generation, grid construction, initial device uploads, and OpenGL rendering are outside the timed calls.

## Build and run on Windows

You’ll need an NVIDIA GPU with CUDA support, the CUDA Toolkit and a compatible NVIDIA driver, Visual Studio 2022 with the **Desktop development with C++** workload, CMake 3.18 or later, and vcpkg. The visualizer requires OpenGL 3.3 core support.

From the repository root in PowerShell, set `$vcpkgRoot` to your vcpkg installation and run:

```powershell
$vcpkgRoot = 'C:/path/to/vcpkg'
& "$vcpkgRoot/vcpkg.exe" install glfw3 glad glm --triplet x64-windows

cmake -S code -B build -G "Visual Studio 17 2022" -A x64 "-DCMAKE_TOOLCHAIN_FILE=$vcpkgRoot/scripts/buildsystems/vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build build --config Release
.\build\Release\culling_benchmark.exe
```

Pass the toolchain path when configuring a fresh build directory because the project’s default points to a local vcpkg installation. You can also open the `code` folder as a CMake project in Visual Studio, set `CMAKE_TOOLCHAIN_FILE` to your vcpkg toolchain, and build and run `culling_benchmark` in x64 Release mode.

## How the visualizer works

Every frame, the app extracts the current camera frustum and runs the grid-based CUDA culling pass. It copies the visibility flags to host memory, then uploads them to an OpenGL buffer with `glBufferSubData`. The grass shader uses those flags during instanced rendering.

## Source layout

- [`code/src/main.cpp`](code/src/main.cpp) generates the scene, builds the grid, runs the benchmark, and renders the visualizer.
- [`code/src/culling.cu`](code/src/culling.cu) implements both CUDA culling paths and handles device memory transfers.
- [`code/include/culling.cuh`](code/include/culling.cuh) defines the shared bounding-box, chunk, and frustum types.
