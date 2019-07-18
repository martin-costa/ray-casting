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

  lightRadius = 700;
  lightIntensity = 255;
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

  lightRays = std::vector<sf::VertexArray>(lights.size());

  //for each light in lights
  for (int j = 0; j < lights.size(); j++) {

    lightRays[j] = sf::VertexArray(sf::TriangleFan, rayCount + 2);

    lightRays[j][0].color = sf::Color(lightIntensity, lightIntensity, lightIntensity);
    lightRays[j][0].position = lights[j].pos;

    //cast rays in circle
    for (int i = 0; i < rayCount; i++) {

      sf::Vector2f dir = sf::Vector2f(cos(4 * asin(1) * i / rayCount + 0.001), sin(4 * asin(1) * i / rayCount + 0.001));
      int t = getClosestIntersection(lightRadius, dir, lights[j]);

      int colorScale = lightIntensity*(lightRadius - t) / lightRadius;
      lightRays[j][i + 1].color = sf::Color(colorScale, colorScale, colorScale);
      lightRays[j][i + 1].position = sf::Vector2f(dir.x * t + lights[j].pos.x, dir.y * t + lights[j].pos.y);
    }
    lightRays[j][rayCount + 1] = lightRays[j][1];
  }
}

double Scene::getClosestIntersection(double t, sf::Vector2f dir, Light light) {

  for (Obstacle obstacle : obstacles) {
    sf::VertexArray lines = obstacle.getVertexArray();

    //loop over all the lines
    for (int i = 0; i < lines.getVertexCount(); i += 2) {

      sf::Vector2f u2 = sf::Vector2f(lines[i + 1].position.x - lines[i].position.x, lines[i + 1].position.y - lines[i].position.y);
      float t2 = (dir.x * (lines[i].position.y - light.pos.y) + dir.y * (light.pos.x - lines[i].position.x)) / (u2.x * dir.y - u2.y * dir.x);

      if (0 < t2 && t2 < 1) {
        float t1 = (lines[i].position.x + u2.x * t2 - light.pos.x) / dir.x;
        if (t1 > 0 && t1 < t) t = t1;
      }

    }
  }
  return t;
}

void Scene::drawScene(sf::RenderWindow* window) {
  castRays(5000);

  for (sf::VertexArray rays : lightRays) {
    //(*window).draw(rays, sf::BlendAdd);

    (*window).draw(rays, sf::BlendMode::BlendMode(sf::BlendMode::SrcColor, sf::BlendMode::OneMinusSrcColor));
  }

  for (Obstacle obs : obstacles) {
    (*window).draw(obs.getVertexArray());
  }
}

void Scene::scaleRadius(double x) {
  lightRadius *= x;
}

void Scene::reset() {
  obstacles = std::vector<Obstacle>();
  addObstacle();

  lights = std::vector<Light>();
  addLight(sf::Vector2f(0, 0));
}