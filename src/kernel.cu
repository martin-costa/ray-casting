#include "kernel.cuh"

//reduce the vector with parallel reduction
__device__ void parrallelReduction(float* x, int size) {
  int n = exp2(ceil(log2f(size)));

  for (int i = n; i > 0; i /= 2) {
    parallelReductionKernel <<< getBlockCount(THREADS, n), THREADS >>> (x, n, i);
  }
}

__global__ void parallelReductionKernel(float* x, int size, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; //index

  if (n + i < size)
    x[i] = fminf(x[i], x[n + i]);
}

//reduce the vector normally
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

  dim3 threadsPerBlock1(1, 1024, 1);
  dim3 numBlocks1(getBlockCount(threadsPerBlock1.x, lightCount), getBlockCount(threadsPerBlock1.y, rayCount), getBlockCount(threadsPerBlock1.z, lineCount));

  //call first kernel to fill the TvalueBlock
  fillTvalueBlockKernel <<< numBlocks1, threadsPerBlock1 >>> (TvalueBlock, castRays, lights, lines, lightCount, rayCount, lineCount, lightRadius);
  
  dim3 threadsPerBlock2(1, 1024);
  dim3 numBlocks2(getBlockCount(threadsPerBlock2.x, lightCount), getBlockCount(threadsPerBlock2.y, rayCount));

  //call second kernel to get correct values
  intersectionReductionKernel <<< numBlocks2, threadsPerBlock2 >>> (TvalueBlock, lightRays, castRays, lights, lines, lightCount, rayCount, lineCount, lightRadius);
}

__host__ std::vector<sf::VertexArray> castRaysAccelerated(std::vector<Light>* lights, sf::VertexArray* lines, float lightRadius, int rayCount) {

  //set up the 2D and 3D host arrays
  int lightCount = lights->size();

  float* lightRaysHost = new float[2 * lightCount * rayCount]();

  float* castRaysHost = new float[rayCount * 2](); //holds directions of circular cast rays [rays][2]
  for (int i = 0; i < rayCount; i++) {
    castRaysHost[i*2 + 0] = cos(2 * PI * i / rayCount + 0.001);
    castRaysHost[i*2 + 1] = sin(2 * PI * i / rayCount + 0.001);
  }

  float* lightsHost = new float[lightCount * 2](); //holds positions of lights [lights][2]
  for (int i = 0; i < lightCount; i++) {
    lightsHost[i*2 + 0] = (*lights)[i].pos.x;
    lightsHost[i*2 + 1] = (*lights)[i].pos.y;
  }

  int lineCount = lines->getVertexCount() / 2;

  float* linesHost = new float[lineCount*4](); //holds positions of lines [lines][4]
  for (int i = 0; i < lineCount; i++) {
    linesHost[i*4 + 0] = (*lines)[i*2].position.x;
    linesHost[i*4 + 1] = (*lines)[i*2].position.y;
    linesHost[i*4 + 2] = (*lines)[i*2 + 1].position.x;
    linesHost[i*4 + 3] = (*lines)[i*2 + 1].position.y;
  }

  //set up the 2D and 3D device arrays for kernels
  float* lightRaysDev = 0;
  cudaMalloc((void**)&lightRaysDev, sizeof(float) * 2 * lightCount * rayCount);

  float* castRaysDev = 0;
  cudaMalloc((void**)&castRaysDev, sizeof(float) * 2 * rayCount);

  float* lightsDev = 0;
  cudaMalloc((void**)&lightsDev, sizeof(float) * 2 * lightCount);

  float* linesDev = 0;
  cudaMalloc((void**)&linesDev, sizeof(float) * 4 * lineCount);

  float* TvalueBlockDev = 0; //will hold onto the t values [lights][rays][lines]
  cudaMalloc((void**)&TvalueBlockDev, sizeof(float) * lightCount * rayCount * lineCount);

  //copy memory from the host to the device
  cudaMemcpy(castRaysDev, castRaysHost, sizeof(float) * 2 * rayCount, cudaMemcpyHostToDevice);
  cudaMemcpy(lightsDev, lightsHost, sizeof(float) * 2 * lightCount, cudaMemcpyHostToDevice);
  cudaMemcpy(linesDev, linesHost, sizeof(float) * 4 * lineCount, cudaMemcpyHostToDevice);

  //invoke the casting kernels
  callCastingKernels(TvalueBlockDev, lightRaysDev, castRaysDev, lightsDev, linesDev, lightCount, rayCount, lineCount, lightRadius);

  //copy the memory back from the device where needed
  cudaMemcpy(lightRaysHost, lightRaysDev, sizeof(float) * 2 * lightCount * rayCount, cudaMemcpyDeviceToHost);

  //free up all the memory on the GPU
  cudaFree(TvalueBlockDev);
  cudaFree(lightRaysDev);
  cudaFree(castRaysDev);
  cudaFree(lightsDev);
  cudaFree(linesDev);

  //put lights into vector of vertex arrays
  std::vector<sf::VertexArray> lightRays = std::vector<sf::VertexArray>((*lights).size());

  for (int i = 0; i < lightCount; i++) {
    lightRays[i] = sf::VertexArray(sf::TriangleFan, rayCount + 2);
    lightRays[i][0].position = (*lights)[i].pos;
    for (int j = 0; j < rayCount; j++) {
      lightRays[i][j + 1].position.x = lightRaysHost[i * rayCount * 2 + j * 2 + 0];
      lightRays[i][j + 1].position.y = lightRaysHost[i * rayCount * 2 + j * 2 + 1];
    }
    lightRays[i][rayCount + 1].position = lightRays[i][1].position;
  }

  //free the host memory
  free(lightRaysHost);
  free(castRaysHost);
  free(lightsHost);
  free(linesHost);

  return lightRays;
}

__host__ std::vector<sf::VertexArray> castRaysUnaccelerated(std::vector<Light>* lights, sf::VertexArray* lines, float lightRadius, int rayCount) {

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

__host__ float getClosestIntersection(float t, sf::VertexArray* lines, sf::Vector2f dir, sf::Vector2f pos) {

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
