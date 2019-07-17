#include "obstacle.hpp"

#include <iostream>

Obstacle::Obstacle(sf::VertexArray lines) {
  this->lines = lines;
}

Obstacle::Obstacle() {}

void Obstacle::addPoint(sf::Vector2f point) {
  lines.resize(lines.getVertexCount() + 2);

  lines[lines.getVertexCount() - 2].color = sf::Color::White;
  lines[lines.getVertexCount() - 1].color = sf::Color::White;


  lines[lines.getVertexCount() - 2].position = (lines.getVertexCount() > 2) ? 
    lines[lines.getVertexCount() - 3].position : point;

  lines[lines.getVertexCount() - 1].position = point;
}

sf::VertexArray Obstacle::getVertexArray() {
  return lines;
}

sf::Vector2f Obstacle::getVertex(int i) {
  return lines[2 * i + i].position;
}