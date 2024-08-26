#version 140

in vec2 position;

uniform float x;
uniform float scale;

void main() {
    gl_Position = vec4(vec2(position.x + x, position.y) * scale, 0.0, 1.0);
}
