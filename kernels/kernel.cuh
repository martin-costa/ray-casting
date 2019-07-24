#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <SFML/Graphics.hpp>
#include <stdio.h>

#include "light.hpp"

#define PI 3.14159265359

//cast rays with kernels
__host__ void
  castRaysAccelerated(float* lightRays, std::vector<Light>* lights, float* linesHost, int lineCount, int rayCount, float lightRadius);

__host__ void
  callCastingKernels(float* TvalueBlock, float* lightRays, float* castRays, float* lights, float* lines,
    int lightCount, int rayCount, int lineCount, float lightRadius);

__global__ void 
  fillTvalueBlockKernel(float* TvalueBlock, float* castRays, float* lights, float* lines, int lightCount,
    int rayCount, int lineCount, float lightRadius);

__global__ void 
  intersectionReductionKernel(float* TvalueBlock, float* lightRays, float* castRays, float* lights, float* lines,
    int lightCount, int rayCount, int lineCount, float lightRadius);

//non reduction
__device__ float
  reduction(float* x, int size);

//aux method to get blocks
__host__ __device__ static int
  getBlockCount(float threadCount, float size) {
    int x = ceil(size / threadCount);
    if (x <= 0) return 1;
    return x;
  }