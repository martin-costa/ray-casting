#include "light.hpp"

Light::Light() {
  this->pos = sf::Vector2f(0, 0);
}

Light::Light(sf::Vector2f pos) {
  this->pos = pos;
}