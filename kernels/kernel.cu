#include "kernel.cuh"

#include <iostream>

//reduce the vector
__device__ float reduction(float* x, int size) {
  int t = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] < t) t = x[i];
  }
  return t;
}

__global__ void intersectionReductionKernel(float* TvalueBlock, float* lightRays, float* castRays, float* lights, float* lines, int lightCount, int rayCount, int lineCount, float lightRadius) {
  int i = threadIdx.x + blockIdx.x * blockDim.x; //the light
  int j = threadIdx.y + blockIdx.y * blockDim.y; //the ray

  if (i < lightCount && j < rayCount) {
    float t = lightRadius;
    if (lineCount > 0) {
      t = reduction(&TvalueBlock[i * rayCount * lineCount + j * lineCount + 0], lineCount);
    }
    lightRays[i * rayCount * 2 + j * 2 + 0] = castRays[j * 2 + 0] * t + lights[i * 2 + 0];
    lightRays[i * rayCount * 2 + j * 2 + 1] = castRays[j * 2 + 1] * t + lights[i * 2 + 1];
  }
}

__global__ void fillTvalueBlockKernel(float* TvalueBlock, float* castRays, float* lights, float* lines, int lightCount, int rayCount, int lineCount, float lightRadius) {
  int i = threadIdx.x + blockIdx.x * blockDim.x; //the light
  int j = threadIdx.y + blockIdx.y * blockDim.y; //the ray
  int k = threadIdx.z + blockIdx.z * blockDim.z; //the line

  if (i < lightCount && j < rayCount && k < lineCount) {
    float ux = lines[k*4 + 2] - lines[k*4 + 0];
    float uy = lines[k*4 + 3] - lines[k*4 + 1];

    float t = (castRays[j*2 + 0] * (lines[k*4 + 1] - lights[i*2 + 1]) + castRays[j*2 + 1] * (lights[i*2 + 0] - lines[k*4 + 0])) / (ux * castRays[j*2 + 1] - uy * castRays[j*2 + 0]);

    if (0 <= t && t <= 1) {
      float t2 = (lines[k * 4 + 0] + ux * t - lights[i * 2 + 0]) / castRays[j * 2 + 0];
      if (0 <= t2) {
        TvalueBlock[i * rayCount * lineCount + j * lineCount + k] = t2;
        return;
      }
    }
    TvalueBlock[i * rayCount * lineCount + j * lineCount + k] = lightRadius;
  }
}

__host__ void callCastingKernels(float* TvalueBlock, float* lightRays, float* castRays, float* lights, float* lines, int lightCount, int rayCount, int lineCount, float lightRadius) {

  dim3 threadsPerBlock1(4, 16, 16); //4 16 16
  dim3 numBlocks1(getBlockCount(threadsPerBlock1.x, lightCount), getBlockCount(threadsPerBlock1.y, rayCount), getBlockCount(threadsPerBlock1.z, lineCount));

  //call first kernel to fill the TvalueBlock
  fillTvalueBlockKernel <<< numBlocks1, threadsPerBlock1 >>> (TvalueBlock, castRays, lights, lines, lightCount, rayCount, lineCount, lightRadius);
  
  dim3 threadsPerBlock2(16, 64); //16 64
  dim3 numBlocks2(getBlockCount(threadsPerBlock2.x, lightCount), getBlockCount(threadsPerBlock2.y, rayCount));

  //call second kernel to get correct values
  intersectionReductionKernel <<< numBlocks2, threadsPerBlock2 >>> (TvalueBlock, lightRays, castRays, lights, lines, lightCount, rayCount, lineCount, lightRadius);
}

__host__ void castRaysAccelerated(float* lightRays, std::vector<Light>* lights, float* linesHost, int lineCount, int rayCount, float lightRadius) {

  //set up the 2D and 3D host arrays
  int lightCount = lights->size();

  float* castRaysHost = new float[rayCount * 2](); //holds directions of circular cast rays [rays][2]
  for (int i = 0; i < rayCount; i++) {
    castRaysHost[i * 2 + 0] = cos(2 * PI * i / rayCount + 0.001);
    castRaysHost[i * 2 + 1] = sin(2 * PI * i / rayCount + 0.001);
  }

  float* lightsHost = new float[lightCount * 2](); //holds positions of lights [lights][2]
  for (int i = 0; i < lightCount; i++) {
    lightsHost[i * 2 + 0] = (*lights)[i].pos.x;
    lightsHost[i * 2 + 1] = (*lights)[i].pos.y;
  }

  //set up the device arrays for kernels
  float* lightRaysDev = 0;
  cudaMalloc((void**)& lightRaysDev, sizeof(float) * 2 * lightCount * rayCount);

  float* castRaysDev = 0;
  cudaMalloc((void**)& castRaysDev, sizeof(float) * 2 * rayCount);

  float* lightsDev = 0;
  cudaMalloc((void**)& lightsDev, sizeof(float) * 2 * lightCount);

  float* linesDev = 0;
  cudaMalloc((void**)& linesDev, sizeof(float) * 4 * lineCount);

  float* TvalueBlockDev = 0; //will hold onto the t values [lights][rays][lines]
  cudaMalloc((void**)& TvalueBlockDev, sizeof(float) * lightCount * rayCount * lineCount);

  //copy memory from the host to the device
  cudaMemcpy(castRaysDev, castRaysHost, sizeof(float) * 2 * rayCount, cudaMemcpyHostToDevice);
  cudaMemcpy(lightsDev, lightsHost, sizeof(float) * 2 * lightCount, cudaMemcpyHostToDevice);
  cudaMemcpy(linesDev, linesHost, sizeof(float) * 4 * lineCount, cudaMemcpyHostToDevice);

  //invoke the casting kernels
  callCastingKernels(TvalueBlockDev, lightRaysDev, castRaysDev, lightsDev, linesDev, lightCount, rayCount, lineCount, lightRadius);

  //copy the memory back from the device where needed

  for (int i = 0; i < lightCount; i++) {
    lightRays[i * (rayCount + 2) * 2 + 0] = (*lights)[i].pos.x;
    lightRays[i * (rayCount + 2) * 2 + 1] = (*lights)[i].pos.y;

    cudaMemcpy(&lightRays[i * (rayCount + 2) * 2 + 2], &lightRaysDev[i * rayCount * 2], sizeof(float) * 2 * rayCount, cudaMemcpyDeviceToHost);

    lightRays[i * (rayCount + 2) * 2 + (rayCount + 1) * 2 + 0] = lightRays[i * (rayCount + 2) * 2 + 1 * 2 + 0];
    lightRays[i * (rayCount + 2) * 2 + (rayCount + 1) * 2 + 1] = lightRays[i * (rayCount + 2) * 2 + 1 * 2 + 1];
  }

  //free up all the memory on the GPU
  cudaFree(TvalueBlockDev);
  cudaFree(lightRaysDev);
  cudaFree(castRaysDev);
  cudaFree(lightsDev);
  cudaFree(linesDev);

  //free the host memory
  free(castRaysHost);
  free(lightsHost);
}
