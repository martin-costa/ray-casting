#include "kernel.cuh"

#include <iostream>

//puts the smallest element into the first slot
__device__ float parrallelReduction(float* x, int size) {
  int n = exp2(ceil(log2f(size)));

  for (int i = n; i > 0; i /= 2) {
    parallelReductionKernel <<< getBlockCount(THREADS, size), THREADS >>> (x, size, i);
  }
  return x[0];
}

__global__ void parallelReductionKernel(float* x, int size, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; //index

  if (n + i < size)
    x[i] = fminf(x[i], x[n + i]);
}

__global__ void intersectionReductionKernel(float*** TvalueBlock, float*** lightRays, float** castRays, float** lights, float** lines, int lightCount, int rayCount, int lineCount) {
  int i = threadIdx.x + blockIdx.x * blockDim.x; //the light
  int j = threadIdx.y + blockIdx.y * blockDim.y; //the ray

  if (i >= lightCount || j >= rayCount) return; //if not in the T value block

  parrallelReduction(TvalueBlock[i][j], lineCount);

  float t = TvalueBlock[i][j][0];

  lightRays[i][j][0] = castRays[j][0] * t + lights[i][0];
  lightRays[i][j][1] = castRays[j][1] * t + lights[i][1];
}

__global__ void fillTvalueBlockKernel(float*** TvalueBlock, float*** lightRays, float** castRays, float** lights, float** lines, int lightCount, int rayCount, int lineCount, float lightRadius) {
  int i = threadIdx.x + blockIdx.x * blockDim.x; //the light
  int j = threadIdx.y + blockIdx.y * blockDim.y; //the ray
  int k = threadIdx.z + blockIdx.z * blockDim.z; //the line

  if (i >= lightCount || j >= rayCount || k >= lineCount) return; //if not in the T value block

  float ux = lines[k][3] - lines[k][0];
  float uy = lines[k][2] - lines[k][1];

  float t2 = (castRays[j][0] * (lines[k][1] - lights[i][1]) + castRays[j][1] * (lights[i][0] - lines[k][0])) / (ux * castRays[j][1] - uy * castRays[j][0]);

  if (0 <= t2 && t2 <= 1) TvalueBlock[i][j][k] = (lights[i][0] + ux * t2 - lights[i][0]) / castRays[j][0];
  else TvalueBlock[i][j][k] = lightRadius;
}

std::vector<sf::VertexArray> castRaysCuda2(std::vector<Light>* lights, sf::VertexArray* lines, float lightRadius, int rayCount) {

  std::vector<sf::VertexArray> lightRays = std::vector<sf::VertexArray>((*lights).size());

  int lightCount = lights->size();

  float*** lightRaysHost = new float**[lightCount](); //will hold the rays
  for (int i = 0; i < lightCount; i++) {
    lightRaysHost[i] = new float* [rayCount]();
    for (int j = 0; j < rayCount; j++) {
      lightRaysHost[i][j] = new float[2]();
    }      
  }

  float*** lightRaysDev = 0;
  cudaMalloc((void**)& lightRaysDev, sizeof(lightRaysHost));

  float** castRaysHost = new float*[rayCount](); //holds directions of circular cast rays
  for (int i = 0; i < rayCount; i++) {
    castRaysHost[i] = new float[2](); 
    castRaysHost[i][0] = cos(2 * PI * i / rayCount + 0.001);
    castRaysHost[i][1] = sin(2 * PI * i / rayCount + 0.001);
  }

  float** castRaysDev = 0;
  cudaMalloc((void**)& castRaysDev, sizeof(castRaysHost));

  float** lightsHost = new float* [lightCount](); //holds positions of lights
  for (int i = 0; i < lightCount; i++) {
    lightsHost[i] = new float[2]();
    lightsHost[i][0] = (*lights)[i].pos.x;
    lightsHost[i][1] = (*lights)[i].pos.y;
  }

  float** lightsDev = 0;
  cudaMalloc((void**)& lightsDev, sizeof(lightsHost));

  int lineCount = lines->getVertexCount() / 2;

  std::cout << lineCount;

  float** linesHost = new float* [lineCount](); //holds positions of lines
  for (int i = 0; i < lineCount*2; i+=2) {
    linesHost[i] = new float[4]();
    linesHost[i][0] = (*lines)[i].position.x;
    linesHost[i][1] = (*lines)[i].position.y;
    linesHost[i][2] = (*lines)[i + 1].position.x;
    linesHost[i][3] = (*lines)[i + 1].position.y;
  }

  float** linesDev = 0;
  cudaMalloc((void**)& linesDev, sizeof(linesHost));

  float*** TvalueBlockDev = 0;
  cudaMalloc((void**)& TvalueBlockDev, sizeof(float) * lightCount * rayCount * lineCount);

  cudaMemcpy(lightRaysDev, lightRaysHost, sizeof(lightRaysHost), cudaMemcpyHostToDevice);
  cudaMemcpy(castRaysDev, castRaysHost, sizeof(castRaysHost), cudaMemcpyHostToDevice);
  cudaMemcpy(lightsDev, lightsHost, sizeof(lightsHost), cudaMemcpyHostToDevice);
  cudaMemcpy(linesDev, linesHost, sizeof(linesHost), cudaMemcpyHostToDevice);

  //leleleleelellee here go the kernels

  dim3 threadsPerBlock1(2, 64, 8);
  dim3 numBlocks1( getBlockCount(threadsPerBlock1.x, lightCount), getBlockCount(threadsPerBlock1.y, rayCount), getBlockCount(threadsPerBlock1.z, lineCount));

  //fillTvalueBlockKernel <<< numBlocks1, threadsPerBlock1 >>> (TvalueBlockDev, lightRaysDev, castRaysDev, lightsDev, linesDev, lightCount, rayCount, lineCount, lightRadius);
  //cudaDeviceSynchronize();

  dim3 threadsPerBlock2(2, 128, 1);
  dim3 numBlocks2(getBlockCount(threadsPerBlock2.x, lightCount), getBlockCount(threadsPerBlock2.y, rayCount));

  //intersectionReductionKernel <<< numBlocks2, threadsPerBlock2 >>> (TvalueBlockDev, lightRaysDev, castRaysDev, lightsDev, linesDev, lightCount, rayCount, lineCount);
  //cudaDeviceSynchronize();

  cudaMemcpy(lightRaysHost, lightRaysDev, sizeof(lightRaysHost), cudaMemcpyDeviceToHost);

  //free up all the memory on the GPU
  cudaFree(TvalueBlockDev);
  cudaFree(lightRaysDev);
  cudaFree(castRaysDev);
  cudaFree(lightsDev);
  cudaFree(linesDev);

  for (int i = 0; i < lightCount; i++) {
    lightRays[i] = sf::VertexArray(sf::TriangleFan, rayCount + 2);
    lightRays[i][0].position = (*lights)[i].pos;
    for (int j = 0; j < rayCount; j++) {
      lightRays[i][j + 1].position.x = lightRaysHost[i][j][0];
      lightRays[i][j + 1].position.y = lightRaysHost[i][j][1];
    }
    lightRays[i][rayCount + 1] = lightRays[i][1];
  }

  //cudaDeviceReset();

  return lightRays;
}















std::vector<sf::VertexArray> castRaysCuda(std::vector<Light>* lights, sf::VertexArray* lines, float lightRadius, int rayCount) {

  std::vector<sf::VertexArray> lightRays = std::vector<sf::VertexArray>((*lights).size());

  //for each light in lights
  for (int j = 0; j < (*lights).size(); j++) {

    lightRays[j] = sf::VertexArray(sf::TriangleFan, rayCount + 2);
    lightRays[j][0].position = (*lights)[j].pos;

    //cast rays in circle
    for (int i = 0; i < rayCount; i++) {

      sf::Vector2f dir = sf::Vector2f(cos(2 * PI * i / rayCount + 0.001), sin(2 * PI * i / rayCount + 0.001));
      int t = getClosestIntersection(lightRadius, lines, dir, (*lights)[j].pos);

      lightRays[j][i + 1].position = sf::Vector2f(dir.x * t + (*lights)[j].pos.x, dir.y * t + (*lights)[j].pos.y);
    }
    lightRays[j][rayCount + 1] = lightRays[j][1];
  }

  return lightRays;
}

float getClosestIntersection(float t, sf::VertexArray* lines, sf::Vector2f dir, sf::Vector2f pos) {

  int lineCount = (lines->getVertexCount() / 2);

  //loop over all the lines
  for (int i = 0; i < lineCount * 2; i += 2) {

    sf::Vector2f u2 = sf::Vector2f((*lines)[i + 1].position.x - (*lines)[i].position.x, (*lines)[i + 1].position.y - (*lines)[i].position.y);
    float t2 = (dir.x * ((*lines)[i].position.y - pos.y) + dir.y * (pos.x - (*lines)[i].position.x)) / (u2.x * dir.y - u2.y * dir.x);

    if (0 < t2 && t2 < 1) {
      float t1 = ((*lines)[i].position.x + u2.x * t2 - pos.x) / dir.x;
      if (t1 > 0 && t1 < t) t = t1;
    }

  }

  return t;
}
