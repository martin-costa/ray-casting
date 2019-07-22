#include "main.hpp"

int main() {

  launch();

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

    std::cout << "\rvertex count: " << (*scene.getVertexArray()).getVertexCount() << " ";
    framerate(FPS, true);

    //display the window and then clear to black
    window.display();
    window.clear(sf::Color(0, 0, 0));
  }

  return 0;
}

void mainLoop() {

  static sf::Vector2i pos = MouseDetector::pos(&window);
  if (leftMouse.down() && pos != leftMouse.pos()) {
    scene.drawLine(sf::Vector2f(leftMouse.pos()));
    pos = leftMouse.pos();
  }

  if (rightMouse.clicked()) scene.addLight(sf::Vector2f(rightMouse.pos()));

  if (keyK.typed()) scene.newLine();
  if (keyD.typed()) scene.closeLine();

  if (keyR.typed()) scene.reset();

  if (keyT.typed()) {
    if ((*scene.getLight(0)).color == sf::Color::Black)
      (*scene.getLight(0)).color = sf::Color::White;
    else (*scene.getLight(0)).color = sf::Color::Black;
  }

  (*scene.getLight(0)).pos = sf::Vector2f(MouseDetector::pos(&window));

  scene.drawScene(&window);
}

void launch() {
  std::cout << "Ray Casting Simulation\n\n";

  std::cout << "controls:\n\n";
  std::cout << "K: new obstacle\n";
  std::cout << "D: close obstacle\n";
  std::cout << "O: increase light radius\n";
  std::cout << "P: decrease light radius\n";
  std::cout << "R: reset\n\n";
  std::cout << "Left Mouse: draw obstacle\n";
  std::cout << "Right Mouse: add light\n\n";
  std::cout << "Source code: github.com/martin-costa/ray-casting\n\n";
}