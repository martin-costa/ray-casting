#include "main.hpp"

int main() {

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  //set normal coord system
  glOrtho(0, WIDTH, HEIGHT, 0, 0, 1);
  glMatrixMode(GL_MODELVIEW);

  //enable blending
  glEnable(GL_BLEND);
  glBlendFunc(GL_ONE,GL_ONE);

  launch();

  //main loop while window is open
  bool windowIsOpen = true;
  while (windowIsOpen) {

    //poll event event to make window responsive
    sf::Event event;
    while (window.pollEvent(event)) {

      //close window if exit button pressed or if escape is pressed
      if (event.type == sf::Event::Closed || sf::Keyboard::isKeyPressed(sf::Keyboard::Escape)) {
        windowIsOpen = false;
      }
    }

    mainLoop();

    std::cout << "\rlines: " << scene.getLineCount() << " lights: " << scene.getLightCount() << " ";

    //control the programs framerate
    framerate(FPS, true);

    window.display();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  }

  return 0;
}

void mainLoop() {

  static sf::Vector2i pos = MouseDetector::pos(&window);
  static sf::Vector2i lastPos = MouseDetector::pos(&window);

  if (leftMouse.down() && pos != leftMouse.pos()) {
    scene.drawLine(sf::Vector2f(leftMouse.pos()));
    pos = leftMouse.pos();
  }

  if (rightMouse.clicked()) scene.addLight(sf::Vector2f(rightMouse.pos()));

  if (keyK.typed()) scene.newLine();
  if (keyD.typed()) scene.closeLine();

  if (keyR.typed()) scene.reset();

  if (keyT.down()) {
    std::vector<float>* lines = scene.getLines();
    float theta = 0.03;
    for (int i = 0; i < scene.getLineCount()*2; i++) {
      sf::Vector2f v = sf::Vector2f((*lines)[i*2] - WIDTH / 2, (*lines)[i * 2 + 1] - HEIGHT / 2);
      v = sf::Vector2f(cos(theta) * v.x - sin(theta) * v.y, sin(theta) * v.x + cos(theta) * v.y);
      (*lines)[i*2] = WIDTH / 2 + v.x;
      (*lines)[i * 2 + 1] = HEIGHT / 2 + v.y;
    }
    scene.updateLines();
  }

  if (middleMouse.down()) {
    std::vector<float>* lines = scene.getLines();
    sf::Vector2i current = middleMouse.pos();
    for (int i = 0; i < scene.getLineCount() * 2; i++) {
      (*lines)[i * 2] += (current.x - lastPos.x);
      (*lines)[i * 2 + 1] += (current.y - lastPos.y);
    }
    scene.updateLines();
  }

  lastPos = leftMouse.pos();

  scene.drawScene(&window);
}

void launch() {
  std::cout << "Ray Casting Simulation\n\n";
  std::cout << "controls:\n\n";
  std::cout << "K: new obstacle\n";
  std::cout << "D: close obstacle\n";
  std::cout << "T: rotate obstacles\n";
  std::cout << "R: reset\n\n";
  std::cout << "Middle Mouse: move obstacles\n";
  std::cout << "Left Mouse: draw obstacle\n";
  std::cout << "Right Mouse: add light\n\n";
  std::cout << "Source code: github.com/martin-costa/ray-casting\n\n";
}