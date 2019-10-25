#!/bin/bash
glslangValidator -V shader.vert.glsl -o shader.vert.spv
glslangValidator -V shader.frag.glsl -o shader.frag.spv
