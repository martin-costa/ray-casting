#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <SFML/Graphics.hpp>
#include <stdio.h>

#include "light.hpp"

#define PI 3.14159265359

#define THREADS 1024

std::vector<sf::VertexArray> castRaysCuda(std::vector<Light>* lights, sf::VertexArray* lines, float lightRadius, int rayCount);

std::vector<sf::VertexArray> castRaysCuda2(std::vector<Light>* lights, sf::VertexArray* lines, float lightRadius, int rayCount);

float getClosestIntersection(float t, sf::VertexArray* lines, sf::Vector2f dir, sf::Vector2f pos);

//__host__ void castRays();

__global__ void fillTvalueBlockKernel(float*** TvalueBlock, float*** lightRays, float** castRays, float** lights, float** lines, 
  int lightCount, int rayCount, int lineCount, float lightRadius);

__global__ void intersectionReductionKernel(float*** TvalueBlock, float*** lightRays, float** castRays, float** lights, float** lines,
  int lightCount, int rayCount, int lineCount);

__global__ void parallelReductionKernel(float* x, int size, int n);

__device__ float parrallelReduction(float* x, int size);

__device__ __host__ static int getBlockCount(int threadCount, int size) {
  return ceil(size / threadCount);
}