#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <SFML/Graphics.hpp>
#include <stdio.h>

#include "light.hpp"

#define PI 3.14159265359

#define THREADS 64

//cast rays with kernels
__host__ std::vector<sf::VertexArray> 
  castRaysAccelerated(std::vector<Light>* lights, sf::VertexArray* lines, float lightRadius, int rayCount);

__host__ void
  callCastingKernels(float* TvalueBlock, float* lightRays, float* castRays, float* lights, float* lines,
    int lightCount, int rayCount, int lineCount, float lightRadius);

__global__ void 
  fillTvalueBlockKernel(float* TvalueBlock, float* castRays, float* lights, float* lines, int lightCount,
    int rayCount, int lineCount, float lightRadius);

__global__ void 
  intersectionReductionKernel(float* TvalueBlock, float* lightRays, float* castRays, float* lights, float* lines,
    int lightCount, int rayCount, int lineCount, float lightRadius);

//cast rays without kernels
__host__ std::vector<sf::VertexArray> 
  castRaysUnaccelerated(std::vector<Light>* lights, sf::VertexArray* lines, float lightRadius, int rayCount);

__host__ float 
  getClosestIntersection(float t, sf::VertexArray* lines, sf::Vector2f dir, sf::Vector2f pos);

//parallel reduction
__global__ void
  parallelReductionKernel(float* x, int size, int n);

__device__ void
  parrallelReduction(float* x, int size);

//non parallel reduction
__device__ float
  reduction(float* x, int size);

//aux method to get blocks
__device__ __host__ static int
  getBlockCount(float threadCount, float size) {
  int x = ceil(size / threadCount);
  if (x <= 0) return 1;
  return x;
}