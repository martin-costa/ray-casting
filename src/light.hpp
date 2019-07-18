#pragma once

#include <SFML/Graphics.hpp>

//light will hold the information for a light source
class Light {
public:

  //holds position
  sf::Vector2f pos;

  Light();

  Light(sf::Vector2f pos);
};
