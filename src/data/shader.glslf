#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(early_fragment_tests) in;

layout(location = 0) in vec4 in_pos;
layout(location = 1) in vec4 frag_color;
layout(location = 2) in vec4 uv;

layout(location = 0) out vec4 color;

layout(std140, set = 0, binding = 0) uniform Args {
    mat4 proj;
    mat4 view;
};
layout(set = 0, binding = 1) uniform texture2D colormap;
layout(set = 0, binding = 2) uniform sampler colorsampler;


void main() {
    float acc = 0.0;

    vec3 frag_pos = in_pos.xyz / in_pos.w;

    color = texture(sampler2D(colormap, colorsampler), uv.xy);
    //color = vec4(0,0,0,1);
}
