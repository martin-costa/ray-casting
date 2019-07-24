#include "light.hpp"

Light::Light() {
  this->pos = sf::Vector2f(0, 0);
  setColor();
}

Light::Light(sf::Vector2f pos) {
  this->pos = pos;
  setColor();
}

void Light::setColor() {
  int a = rand() % 6;
  if (a == 0) color = sf::Color(255, 160, 160); //red
  if (a == 1) color = sf::Color(160, 255, 160); //green
  if (a == 2) color = sf::Color(160, 160, 255); //blue

  if (a == 3) color = sf::Color(255, 255, 160); //yellow
  if (a == 4) color = sf::Color(255, 160, 255); //magenta
  if (a == 5) color = sf::Color(160, 255, 255); //cyan
}