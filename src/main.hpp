#pragma once

#include <iostream>
#include <SFML/Graphics.hpp>

#include "inputdetector.hpp"
#include "framerate.hpp"
#include "scene.hpp"

//define width and height
#define WIDTH 900
#define HEIGHT 900

//max FPS of the program
#define FPS 60

//main method where all the action starts
int main();

//runs before main program starts
void launch();

//where most of the program goes down
void mainLoop();

//define the window the program will use
sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "2D Ray Casting", sf::Style::Close);

//initialise some key handlers to be used
KeyDetector keyK(sf::Keyboard::K);
KeyDetector keyD(sf::Keyboard::D);
KeyDetector keyR(sf::Keyboard::R);

MouseDetector leftMouse(sf::Mouse::Left, &window);
MouseDetector rightMouse(sf::Mouse::Right, &window);

KeyDetector keyT(sf::Keyboard::T);

//make a scene
Scene scene(WIDTH, HEIGHT);