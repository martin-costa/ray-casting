#pragma once

#include <SFML/Graphics.hpp>
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

  sf::VertexArray lines = sf::VertexArray(sf::Lines, 0);
  std::vector<Light> lights;

  float lightRadius = 10000;

  std::vector<sf::VertexArray> lightRays;

  sf::Vector2f lineStart;

  //get closest obstacle from light in the direction light
  float getClosestIntersection(float t, sf::Vector2f dir, sf::Vector2f pos);

public:
  Scene(int width, int height);

  //returns pointer to the vertex array
  sf::VertexArray* getVertexArray();

  //gets the ith light in the vector
  Light* getLight(int i);

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
  void drawScene(sf::RenderWindow* window);

  //reset elements in scene
  void reset();
};