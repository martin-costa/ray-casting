#include "scene.hpp"

#include <iostream>

Scene::Scene(int width, int height) {
  this->width = width;
  this->height = height;

  sf::VertexArray borders = sf::VertexArray(sf::Lines, 8);

  //set up borders for scene
  borders[0].position = sf::Vector2f(0, 0);
  borders[1].position = sf::Vector2f(0, height + 1);

  borders[2].position = sf::Vector2f(0, height + 1);
  borders[3].position = sf::Vector2f(width + 1, height + 1);

  borders[4].position = sf::Vector2f(width + 1, height + 1);
  borders[5].position = sf::Vector2f(width + 1, 0);

  borders[6].position = sf::Vector2f(width + 1, 0);
  borders[7].position = sf::Vector2f(0, 0);

  obstacles.push_back(Obstacle(borders));

  addObstacle();

  addLight(sf::Vector2f(0, 0));
}

Obstacle* Scene::getObstacle(int i) {
  return &obstacles[i];
}

Light* Scene::getLight(int i) {
  return &lights[i];
}

void Scene::addObstacle() {
  obstacles.push_back(Obstacle());
  obstacleBuffer = getObstacle(obstacles.size() - 1);
}

void Scene::drawToBuffer(sf::Vector2f point) {
  (*obstacleBuffer).addPoint(point);
}

void Scene::closeBuffer() {
  (*obstacleBuffer).addPoint((*obstacleBuffer).getVertex(0));
  addObstacle();
}

void Scene::addLight(sf::Vector2f pos) {
  lights.push_back(Light(pos));
}

void Scene::castRays(int rayCount) {
  lightRays = sf::VertexArray(sf::Lines, rayCount * 2 * lights.size());

  //for each light in lights
  for (int j = 0; j < lights.size(); j++) {

    //cast rays in circle
    for (int i = 0; i < rayCount*2; i+= 2) {
      lightRays[i + j * rayCount * 2].position = lights[j].pos;

      sf::Vector2f m = sf::Vector2f(cos(2 * asin(1) * i / rayCount + 0.001), sin(2 * asin(1) * i / rayCount + 0.001));
      int t = 100000;

      for (Obstacle obstacle : obstacles) {
        sf::VertexArray lines = obstacle.getVertexArray();

        //loop over all the lines
        for (int i = 0; i < lines.getVertexCount(); i+= 2) {

          sf::Vector2f u2 = sf::Vector2f(lines[i + 1].position.x - lines[i].position.x, lines[i + 1].position.y - lines[i].position.y);
          float t2 = (m.x * (lines[i].position.y - lights[j].pos.y) + m.y * (lights[j].pos.x - lines[i].position.x)) / (u2.x * m.y - u2.y * m.x);

          if (0 < t2 && t2 < 1) {
            float t1 = (lines[i].position.x + u2.x * t2 - lights[j].pos.x) / m.x;
            if (t1 > 0 && t1 < t) t = t1;
          }

        }
      }
      lightRays[i + 1 + j * rayCount * 2].position = sf::Vector2f(m.x * t + lights[j].pos.x, m.y * t + lights[j].pos.y);
    }
  }
}

void Scene::drawScene(sf::RenderWindow* window) {
  for (Obstacle obs : obstacles) {
    (*window).draw(obs.getVertexArray());
  }
  castRays(1000);
  (*window).draw(lightRays);
}

void Scene::reset() {
  obstacles = std::vector<Obstacle>();
  addObstacle();

  lights = std::vector<Light>();
  addLight(sf::Vector2f(0, 0));
}