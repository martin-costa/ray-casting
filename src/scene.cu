#include "scene.cuh"

Scene::Scene(int width, int height) {
  this->width = width;
  this->height = height;

  reset();
}

std::vector<float>* Scene::getLines() {
  return &lines;
}

std::vector<float>* Scene::getLights() {
  return &lights;
}

std::vector<sf::Color>* Scene::getLightColors() {
  return &lightColors;
}

void Scene::drawLine(sf::Vector2f point) {
  if (lines.size() % 4 == 0) {
    lines.resize(lines.size() + 2);
    lines[lines.size() - 2] = point.x;
    lines[lines.size() - 1] = point.y;
    lineStart = point;
  }
  else {
    lines.resize(lines.size() + 4);
    lines[lines.size() - 4] = point.x;
    lines[lines.size() - 3] = point.y;
    lines[lines.size() - 2] = point.x;
    lines[lines.size() - 1] = point.y;
  }

  resetLines();
}

void Scene::closeLine() {
  if (lines.size() % 4 == 0) return;
  lines.resize(lines.size() + 2);
  lines[lines.size() - 2] = lineStart.x;
  lines[lines.size() - 1] = lineStart.y;

  resetLines();
}

void Scene::newLine() {
  if (lines.size() % 4 == 0) return;
  lines.resize(lines.size() - 2);

  resetLines();
}

void Scene::addLight(sf::Vector2f pos) {
  lights.push_back(pos.x);
  lights.push_back(pos.y);
  lightColors.push_back(generateColor());
  delete[] lightRays;
  lightRays = new float[(rayCount + 2) * lights.size()]();

  resetLights();
}

sf::Color Scene::generateColor() {
  int a = rand() % 6;
  if (a == 0) return sf::Color(255, 100, 100); //red
  if (a == 1) return sf::Color(100, 255, 100); //green
  if (a == 2) return sf::Color(100, 100, 255); //blue
  if (a == 3) return sf::Color(255, 255, 100); //yellow
  if (a == 4) return sf::Color(255, 100, 255); //magenta
  if (a == 5) return sf::Color(100, 255, 255); //cyan
}

void Scene::drawScene(sf::Window* window) {
  castRays(TvalueBlockDev, lightRaysDev, linesDev, lightsDev, lights.data(), lightRays, castRaysDev, getLightCount(), getLineCount(), rayCount, lightRadius);

  sf::Shader::bind(&lightShader);

  glEnableClientState(GL_VERTEX_ARRAY);
  glVertexPointer(2, GL_FLOAT, 0, lightRays);

  for (int i = 0; i < lights.size() / 2; i++) {
    lightShader.setUniform("pos", sf::Vector2f(lights[i*2], -lights[i*2 + 1] + height));
    lightShader.setUniform("color", sf::Vector3f(lightColors[i].r, lightColors[i].g, lightColors[i].b));

    glDrawArrays(GL_TRIANGLE_FAN, i * (rayCount + 2), (rayCount + 2));
  }

  sf::Shader::bind(&lineShader);

  glVertexPointer(2, GL_FLOAT, 0, lines.data());
  glDrawArrays(GL_LINES, 0, lines.size() / 2);

  glDisableClientState(GL_VERTEX_ARRAY);
}

void Scene::reset() {

  //create the vector of lines
  this->lines = std::vector<float>(0);

  //create the vector of lights
  this->lights = std::vector<float>(0);

  this->lightColors = std::vector<sf::Color>(0);

  delete[] lightRays;
  this->lightRays = new float[(rayCount + 2) * lights.size() * 2]();

  lightShader.loadFromFile("lightShader.fs", sf::Shader::Fragment);
  lineShader.loadFromFile("lineShader.fs", sf::Shader::Fragment);

  //init cast rays on device
  float* castRays = new float[rayCount*2]();
  for (int i = 0; i < rayCount; i++) {
    castRays[i * 2 + 0] = cos(2 * PI * i / rayCount + 0.001);
    castRays[i * 2 + 1] = sin(2 * PI * i / rayCount + 0.001);
  }
  cudaMalloc((void**)& castRaysDev, sizeof(float) * 2 * rayCount);
  cudaMemcpy(castRaysDev, castRays, sizeof(float) * 2 * rayCount, cudaMemcpyHostToDevice);

  free(castRays);
}

void Scene::resetLights() {
  cudaFree(lightsDev);
  cudaMalloc((void**)& lightsDev, sizeof(float) * 2 * getLightCount());
  cudaMemcpy(lightsDev, lights.data(), sizeof(float) * 2 * getLightCount(), cudaMemcpyHostToDevice);

  cudaFree(lightRaysDev);
  cudaMalloc((void**)& lightRaysDev, sizeof(float) * 2 * getLightCount() * rayCount);

  resetTvalueBlock();
}

void Scene::resetLines() {
  cudaFree(linesDev);
  cudaMalloc((void**)& linesDev, sizeof(float) * 4 * getLineCount());
  cudaMemcpy(linesDev, lines.data(), sizeof(float) * 4 * getLineCount(), cudaMemcpyHostToDevice);

  resetTvalueBlock();
}

void Scene::resetTvalueBlock() {
  cudaFree(TvalueBlockDev);
  cudaMalloc((void**)& TvalueBlockDev, sizeof(float) * getLightCount() * rayCount * getLineCount());
}

void Scene::updateLights() {
  cudaMemcpy(lightsDev, lights.data(), sizeof(float) * 2 * getLightCount(), cudaMemcpyHostToDevice);
}

void Scene::updateLines() {
  cudaMemcpy(linesDev, lines.data(), sizeof(float) * 4 * getLineCount(), cudaMemcpyHostToDevice);
}

/*
 * methods for raycasting
 */

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
    float ux = lines[k * 4 + 2] - lines[k * 4 + 0];
    float uy = lines[k * 4 + 3] - lines[k * 4 + 1];

    float t = (castRays[j * 2 + 0] * (lines[k * 4 + 1] - lights[i * 2 + 1]) + castRays[j * 2 + 1] * (lights[i * 2 + 0] - lines[k * 4 + 0])) / (ux * castRays[j * 2 + 1] - uy * castRays[j * 2 + 0]);

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
  fillTvalueBlockKernel << < numBlocks1, threadsPerBlock1 >> > (TvalueBlock, castRays, lights, lines, lightCount, rayCount, lineCount, lightRadius);

  dim3 threadsPerBlock2(16, 64); //16 64
  dim3 numBlocks2(getBlockCount(threadsPerBlock2.x, lightCount), getBlockCount(threadsPerBlock2.y, rayCount));

  //call second kernel to get correct values
  intersectionReductionKernel << < numBlocks2, threadsPerBlock2 >> > (TvalueBlock, lightRays, castRays, lights, lines, lightCount, rayCount, lineCount, lightRadius);
}

__host__ void Scene::castRays(float* TvalueBlockDev, float* lightRaysDev, float* linesDev, float* lightsDev, float* lights, float* lightRays, float* castRaysDev, int lightCount, int lineCount, int rayCount, float lightRadius) {

  //invoke the casting kernels
  callCastingKernels(TvalueBlockDev, lightRaysDev, castRaysDev, lightsDev, linesDev, lightCount, rayCount, lineCount, lightRadius);

  //copy the memory back from the device
  for (int i = 0; i < lightCount; i++) {
    lightRays[i * (rayCount + 2) * 2 + 0] = lights[i * 2 + 0];
    lightRays[i * (rayCount + 2) * 2 + 1] = lights[i * 2 + 1];

    cudaMemcpy(&lightRays[i * (rayCount + 2) * 2 + 2], &lightRaysDev[i * rayCount * 2], sizeof(float) * 2 * rayCount, cudaMemcpyDeviceToHost);

    lightRays[i * (rayCount + 2) * 2 + (rayCount + 1) * 2 + 0] = lightRays[i * (rayCount + 2) * 2 + 1 * 2 + 0];
    lightRays[i * (rayCount + 2) * 2 + (rayCount + 1) * 2 + 1] = lightRays[i * (rayCount + 2) * 2 + 1 * 2 + 1];
  }
}