#version 430
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec3 aColor;

out vec2 TexCoords;
out vec3 WorldPos;
out vec3 Normal;
out vec3 Color;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main()
{
    TexCoords = vec2(0, 0);
    WorldPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(model) * aNormal;   
    Color = aColor;

    gl_Position =  projection * view * vec4(WorldPos, 1.0);
}