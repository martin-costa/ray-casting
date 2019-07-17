#pragma once

#include <SFML/Graphics.hpp>

//obstacle will hold the object to be drawn as well
//as the information of where the obstacle is
class Obstacle {
private:
  sf::VertexArray lines = sf::VertexArray(sf::Lines, 0);

public:
  //set obstacle equal to a vertex array
  Obstacle(sf::VertexArray lines);

  //initialise a new empty obstacle
  Obstacle();

  //add a point to the obstacle
  void addPoint(sf::Vector2f point);

  //get the vertex array by value
  sf::VertexArray getVertexArray();

  //get vertex
  sf::Vector2f getVertex(int i);
};