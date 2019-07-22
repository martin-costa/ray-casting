#include "scene.hpp"

Scene::Scene(int width, int height) {
  this->width = width;
  this->height = height;

  reset();
}

sf::VertexArray* Scene::getVertexArray() {
  return &lines;
}

Light* Scene::getLight(int i) {
  return &lights[i];
}

void Scene::drawLine(sf::Vector2f point) {
  if (lines.getVertexCount() % 2 == 0) {
    lines.resize(lines.getVertexCount() + 1);
    lines[lines.getVertexCount() - 1].position = point;
    lineStart = point;
    return;
  }
  lines.resize(lines.getVertexCount() + 2);
  lines[lines.getVertexCount() - 2].position = point;
  lines[lines.getVertexCount() - 1].position = point;
}

void Scene::closeLine() {
  if (lines.getVertexCount() % 2 == 0) return;
  lines.resize(lines.getVertexCount() + 1);
  lines[lines.getVertexCount() - 1].position = lineStart;
}

void Scene::newLine() {
  if (lines.getVertexCount() % 2 == 0) return;
  lines.resize(lines.getVertexCount() - 1);
}

void Scene::addLight(sf::Vector2f pos) {
  lights.push_back(Light(pos));
}

void Scene::castRays(int rayCount) {
  static bool gpuOn = true;
  static KeyDetector keyL(sf::Keyboard::L);
  if (keyL.typed()) gpuOn = !gpuOn;

  if(gpuOn)
    lightRays = castRaysAccelerated(&lights, &lines, lightRadius, rayCount);
  else
    lightRays = castRaysUnaccelerated(&lights, &lines, lightRadius, rayCount);
}

void Scene::drawScene(sf::RenderWindow* window) {
  castRays(2048); //a multiple of 1024 (ideally 4096) is suggested

  sf::Shader shader;
  shader.loadFromFile("shader.fs", sf::Shader::Fragment);

  for (int i = 0; i < lights.size(); i++) {
    sf::Vector2f pos(lights[i].pos.x, -lights[i].pos.y + height);

    shader.setUniform("pos", pos);
    shader.setUniform("color", sf::Vector3f(lights[i].color.r,lights[i].color.g, lights[i].color.b));

    (*window).draw(lightRays[i], sf::RenderStates(sf::BlendAdd, sf::Transform::Identity, NULL, &shader));
  }

  (*window).draw(lines);
}

void Scene::reset() {

  //set up borders for scene
  this->lines = sf::VertexArray(sf::Lines, 0);

  //set up lights
  this->lights = std::vector<Light>();
  addLight(sf::Vector2f(0, 0));
  lights[0].color = sf::Color::Black;
}