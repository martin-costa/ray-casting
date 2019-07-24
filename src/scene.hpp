#pragma once

#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>

#include <vector> 
#include <cmath>
#include <cstdlib>

#include "light.hpp"
#include "kernel.cuh"

#include "inputdetector.hpp"

#define PI 3.14159265359

class Scene {
private:
  int width;
  int height;

  std::vector<float> lines = std::vector<float>(0); //i*4 + j
  std::vector<Light> lights;

  float* lightRays;

  float lightRadius = 10000;
  int rayCount = 2048;

  sf::Vector2f lineStart;

  sf::Shader lightShader;
  sf::Shader lineShader;

public:
  Scene(int width, int height);

  //draw to last obstacle in obstacles
  void drawLine(sf::Vector2f point);

  //make currect drawing a closed loop
  void newLine();

  //go onto new drawing
  void closeLine();

  //add a new obstacle
  void addLight(sf::Vector2f pos);

  //cast rays from all light sources
  void castRays(int rayCount);

  //pass in ref. to window and draw scene there
  void drawScene(sf::Window* window);

 /*
  * setters and getters
  */

  //returns pointer to the lines
  std::vector<float>* getLines();

  //returns pointer to the lights
  std::vector<Light>* getLights();

  //get line count
  int getLineCount() { return lines.size() / 4; }

  //get light count
  int getLightCount() { return lights.size(); }

  //reset elements in scene
  void reset();
};