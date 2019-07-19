#pragma once

#include <SFML/Graphics.hpp>

//light will hold the information for a light source
class Light {
public:

  //holds position
  sf::Vector2f pos;

  Light() {
    this->pos = sf::Vector2f(0, 0);
  }

  Light(sf::Vector2f pos) {
    this->pos = pos;
  }
};