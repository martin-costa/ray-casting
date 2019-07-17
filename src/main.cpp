#include "main.hpp"

int main() {

  //main loop while window is open
  while (window.isOpen()) {

    //poll event event to make window responsive
    sf::Event event;
    while (window.pollEvent(event)) {

      //close window if exit button pressed or if escape is pressed
      if (event.type == sf::Event::Closed || sf::Keyboard::isKeyPressed(sf::Keyboard::Escape)) {
        window.close();
      }
    }

    mainLoop();

    //control the programs framerate
    framerate(FPS, true);

    //display the window and then clear to black
    window.display();
    window.clear(sf::Color(0, 0, 0));
  }

  return 0;
}

void mainLoop() {

  if (leftMouse.down()) scene.drawToBuffer(sf::Vector2f(leftMouse.pos()));

  if (rightMouse.clicked()) scene.addLight(sf::Vector2f(rightMouse.pos()));

  if (keyK.typed()) scene.addObstacle();

  if (keyD.typed()) scene.closeBuffer();

  if (keyR.typed()) scene.reset();

  (*scene.getLight(0)).pos = sf::Vector2f(leftMouse.pos());

  scene.drawScene(&window);
}