uniform vec2 pos;

uniform vec3 color = vec3(255.0, 255.0, 255.0);

uniform float intensity = 15.0;

void main() {

  float distance = length(pos - gl_FragCoord.xy)/intensity;

  gl_FragColor = vec4(color/(distance*255.0), 1.0);
}