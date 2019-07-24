#include "scene.hpp"

Scene::Scene(int width, int height) {
  this->width = width;
  this->height = height;

  reset();
}

std::vector<float>* Scene::getLines() {
  return &lines;
}

std::vector<Light>* Scene::getLights() {
  return &lights;
}

void Scene::drawLine(sf::Vector2f point) {
  //if line incomplete
  if (lines.size() % 4 == 0) {
    lines.resize(lines.size() + 2);
    lines[lines.size() - 2] = point.x;
    lines[lines.size() - 1] = point.y;
    lineStart = point;
    return;
  }

  //if line complete
  lines.resize(lines.size() + 4);
  lines[lines.size() - 4] = point.x;
  lines[lines.size() - 3] = point.y;
  lines[lines.size() - 2] = point.x;
  lines[lines.size() - 1] = point.y;
}

void Scene::closeLine() {
  if (lines.size() % 4 == 0) return;
  lines.resize(lines.size() + 2);
  lines[lines.size() - 2] = lineStart.x;
  lines[lines.size() - 1] = lineStart.y;
}

void Scene::newLine() {
  if (lines.size() % 4 == 0) return;
  lines.resize(lines.size() - 2);
}

void Scene::addLight(sf::Vector2f pos) {
  lights.push_back(Light(pos));
}

void Scene::castRays(int rayCount) {
  lightRays = new float[(rayCount + 2)*lights.size()*2]();
  castRaysAccelerated(lightRays, &lights, lines.data(), lines.size() / 4, rayCount, lightRadius);
}

void Scene::drawScene(sf::Window* window) {
  castRays(rayCount); //a multiple of 1024 (ideally 4096) is suggested

  sf::Shader::bind(&lightShader);

  glEnableClientState(GL_VERTEX_ARRAY);
  glVertexPointer(2, GL_FLOAT, 0, lightRays);

  for (int i = 0; i < lights.size(); i++) {
    lightShader.setUniform("pos", sf::Vector2f(lights[i].pos.x, -lights[i].pos.y + height));
    lightShader.setUniform("color", sf::Vector3f(lights[i].color.r, lights[i].color.g, lights[i].color.b));

    glDrawArrays(GL_TRIANGLE_FAN, i * (rayCount + 2), (rayCount + 2));
  }

  sf::Shader::bind(&lineShader);

  glVertexPointer(2, GL_FLOAT, 0, lines.data());
  glDrawArrays(GL_LINES, 0, lines.size() / 2);

  glDisableClientState(GL_VERTEX_ARRAY);
}

void Scene::reset() {

  this->lines = std::vector<float>(0);

  //set up lights
  this->lights = std::vector<Light>(0);

  lightShader.loadFromFile("lightShader.fs", sf::Shader::Fragment);
  lineShader.loadFromFile("lineShader.fs", sf::Shader::Fragment);
}