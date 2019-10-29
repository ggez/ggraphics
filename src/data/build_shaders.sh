#!/bin/bash
glslangValidator -V quad.vert.glsl -o quad.vert.spv
glslangValidator -V quad.frag.glsl -o quad.frag.spv

glslangValidator -V mesh.vert.glsl -o mesh.vert.spv
glslangValidator -V mesh.frag.glsl -o mesh.frag.spv
