#include "inputdetector.hpp"

//defining the KeyDetector methods
KeyDetector::KeyDetector(sf::Keyboard::Key key) {
  this->key = key;
}

//method to update type and get if just typed
bool KeyDetector::typed() {
  if (sf::Keyboard::isKeyPressed(key) && checkKey) {
    checkKey = false;
    return true;
  }
  if (!sf::Keyboard::isKeyPressed(key)) checkKey = true;
  return false;
}

//method to get if key down
bool KeyDetector::down() {
  return sf::Keyboard::isKeyPressed(key);
}

//defining the MouseDetector methods
MouseDetector::MouseDetector(sf::Mouse::Button button) {
  this->button = button;
}

MouseDetector::MouseDetector(sf::Window* window) {
  this->window = window;
}

MouseDetector::MouseDetector(sf::Mouse::Button button, sf::Window *window) {
  this->button = button;
  this-> window = window;
}

//method to update click and get if just clicked
bool MouseDetector::clicked() {
  if (sf::Mouse::isButtonPressed(button) && checkButton) {
    checkButton = false;
    return true;
  }
  if (!sf::Mouse::isButtonPressed(button)) checkButton = true;
  return false;
}

//method to get if key down
bool MouseDetector::down() {
  return sf::Mouse::isButtonPressed(button);
}

//method to get mouse position
sf::Vector2i MouseDetector::pos() {
  return sf::Mouse::getPosition(*window);
}

//static method to get mouse position
sf::Vector2i MouseDetector::pos(sf::Window *window) {
  return sf::Mouse::getPosition(*window);
}