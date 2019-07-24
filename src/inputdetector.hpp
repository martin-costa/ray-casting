#pragma once

#include <SFML/Graphics.hpp>
#include <iostream>

//class to handle keys
class KeyDetector {
private:
  bool checkKey = true;

  //store key we are checking
  sf::Keyboard::Key key;

public:
  KeyDetector(sf::Keyboard::Key key);

  //method to update type and get if just typed
  bool typed();

  //method to get if key down
  bool down();
};

//class to handle mouse
class MouseDetector {
private:
  bool checkButton = true;

  //store key we are checking
  sf::Mouse::Button button;

  //can store window in needed
  sf::Window *window;

public:
  MouseDetector(sf::Mouse::Button button);

  MouseDetector(sf::Window* window);

  MouseDetector(sf::Mouse::Button button, sf::Window* window);

  //method to update click and get if just clicked
  bool clicked();

  //method to get if key down
  bool down();

  //method to get mouse position
  sf::Vector2i pos();

  //static method to get mouse position
  static sf::Vector2i pos(sf::Window *window);
};