#pragma once

#include <iostream>
#include <SFML/Graphics.hpp>

#include "clickdetector.hpp"
#include "framerate.hpp"
#include "scene.hpp"

//main method where all the action starts
int main();

//where most of the program goes down
void mainLoop();

//define the (max) fps the program will run at
int FPS = 60;

//define size of window and scene
const int WIDTH = 900;
const int HEIGHT = 600;

//define the window the program will use
sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Ray Casting", sf::Style::Close);

//initialise some key handlers to be used
KeyDetector keyK(sf::Keyboard::K);
KeyDetector keyD(sf::Keyboard::D);
KeyDetector keyR(sf::Keyboard::R);

MouseDetector leftMouse(sf::Mouse::Left, &window);
MouseDetector rightMouse(sf::Mouse::Right, &window);

//make a scene
Scene scene(WIDTH, HEIGHT);