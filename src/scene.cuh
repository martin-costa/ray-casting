#pragma once

#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define PI 3.14159265359

class Scene {
private:
  int width;
  int height;

  float lightRadius = 10000;
  int rayCount = 2048; //a multiple of 1024 (ideally 2048 or 4096) is suggested

  std::vector<float> lines; //i*4 + j
  std::vector<float> lights;
  std::vector<sf::Color> lightColors;

  //arrays on the device
  float* lightRaysDev = 0;
  float* castRaysDev = 0;
  float* lightsDev = 0;
  float* linesDev = 0;

  float* TvalueBlockDev = 0;

  //array to copy rays into
  float* lightRays;

  //start of line currently being drawn
  sf::Vector2f lineStart;

  //shaders
  sf::Shader lightShader;
  sf::Shader lineShader;

  sf::Color generateColor();

  void resetTvalueBlock();

  void resetLights();
  void resetLines();

public:
  Scene(int width, int height);

  void updateLights();
  void updateLines();

  //draw to last obstacle in obstacles
  void drawLine(sf::Vector2f point);

  //make currect drawing a closed loop
  void newLine();

  //go onto new drawing
  void closeLine();

  //add a new obstacle
  void addLight(sf::Vector2f pos);

  //cast rays from all light sources
  void castRays(float* TvalueBlockDev, float* lightRaysDev, float* linesDev, float* lightsDev, float* lights, float* lightRays, float* castRaysDev, int lightCount, int lineCount, int rayCount, float lightRadius);

  //pass in ref. to window and draw scene there
  void drawScene(sf::Window* window);

  /*
   * setters and getters
   */

   //returns pointer to the lines
  std::vector<float>* getLines();

  //returns pointer to the light positions
  std::vector<float>* getLights();

  //returns pointer to the light colours
  std::vector<sf::Color>* getLightColors();

  //get line count
  int getLineCount() { return lines.size() / 4; }

  //get light count
  int getLightCount() { return lights.size() / 2; }

  //reset elements in scene
  void reset();
};

/*
 * methods for raycasting
 */

//kernels for ray casting
__global__ void
fillTvalueBlockKernel(float* TvalueBlock, float* castRays, float* lights, float* lines, int lightCount,
  int rayCount, int lineCount, float lightRadius);

__global__ void
intersectionReductionKernel(float* TvalueBlock, float* lightRays, float* castRays, float* lights, float* lines,
  int lightCount, int rayCount, int lineCount, float lightRadius);

//method to call the kernels
__host__ void
callCastingKernels(float* TvalueBlock, float* lightRays, float* castRays, float* lights, float* lines,
  int lightCount, int rayCount, int lineCount, float lightRadius);

//get smallest elt in vector
__device__ float
 reduction(float* x, int size);

//aux method to get size block
__host__ __device__ static int
getBlockCount(float threadCount, float size) {
  int x = ceil(size / threadCount);
  if (x <= 0) return 1;
  return x;
}