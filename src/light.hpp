#pragma once

#include <SFML/Graphics.hpp>

//light will hold the information for a light source
class Light {
public:

  //holds position
  sf::Vector2f pos;

  //holds color
  sf::Color color = sf::Color::White;

  Light();

  Light(sf::Vector2f pos);
};
