#pragma once

#include <SFML/Graphics.hpp>
#include <vector> 
#include <cmath>
#include <cstdlib>

#define PI 3.14159265359

//light will hold the information for a light source
class Light {
public:

  //holds position
  sf::Vector2f pos;

  //hold color
  sf::Color color;

  Light();

  Light(sf::Vector2f pos);

  void setColor();
};

class Scene {
private:
  int width;
  int height;

  sf::VertexArray lines = sf::VertexArray(sf::Lines, 0);;
  std::vector<Light> lights;

  double lightRadius = 10000;

  std::vector<sf::VertexArray> lightRays;

  sf::Vector2f lineStart;

  //get closest obstacle from light in the direction light
  double getClosestIntersection(double t, sf::Vector2f dir, Light light);

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