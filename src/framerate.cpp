#include "framerate.hpp"

//define the framerate method
void framerate(int FPS, bool show) {

  static sf::Clock clock;
  sf::Time elapsed = clock.getElapsedTime();
  sf::Time TPF = sf::seconds(1.0f / FPS);
  sf::Time sleeptime = sf::seconds(0);

  if (TPF > elapsed) {
    sleeptime = TPF - elapsed;
    sf::sleep(sleeptime);
  }

  static int counter = 0;
  if (show && counter++ > 10) { //output every 10 frames
    std::cout << "framerate: " << 1 / (elapsed + sleeptime).asSeconds() << "\r";
    counter = 0;
  }
  clock.restart();
}