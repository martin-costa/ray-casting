#pragma once

#include <SFML/Graphics.hpp>
#include <vector> 
#include <cmath>

#include "obstacle.hpp"
#include "light.hpp"

//double PI = asin(1);

class Scene {
private:
  int width;
  int height;

  std::vector<Obstacle> obstacles;
  std::vector<Light> lights;

  double lightIntensity;
  double lightRadius;

  std::vector<sf::VertexArray> lightRays;

  Obstacle* obstacleBuffer;

  //get closest obstacle from light in the direction light
  double getClosestIntersection(double t, sf::Vector2f dir, Light light);

public:
  Scene(int width, int height);

  //gets the ith obstacle in the vector
  Obstacle* getObstacle(int i);

  //gets the ith light in the vector
  Light* getLight(int i);

  //add a new obstacle
  void addObstacle();

  //draw to last obstacle in obstacles
  void drawToBuffer(sf::Vector2f point);

  //make currect buffer a closed loop
  void closeBuffer();

  //add a new obstacle
  void addLight(sf::Vector2f pos);

  //cast rays from all light sources
  void castRays(int rayCount);

  //pass in ref. to window and draw scene there
  void drawScene(sf::RenderWindow* window);

  //multiply light radius
  void scaleRadius(double x);

  //reset elements in scene
  void reset();
};